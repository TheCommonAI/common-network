"""Planner tests. Offline, no model weights needed.

Run from the repository root:
    python3 -m unittest pedal.tests.test_plan -v
"""

from __future__ import annotations

import unittest

from pedal.plan import (
    GiB,
    MLX4_LAYER_BYTES,
    NF4_LAYER_BYTES,
    PlanError,
    Member,
    kv_bytes_per_token_per_layer,
    plan_qwen3_32b,
    plan_split,
)

KV = kv_bytes_per_token_per_layer(8, 128)  # Qwen3-32B: 4096 bytes per token per layer


class Qwen32BPlan(unittest.TestCase):
    def test_real_budgets_fit_at_2048_context(self):
        plan = plan_qwen3_32b(11.0, 6.5, context_tokens=2048)
        self.assertEqual(plan.route(), [("mac", 0, 38), ("pc", 38, 64)])
        mac, pc = plan.assignments
        self.assertGreaterEqual(mac.headroom_bytes, 0)
        self.assertGreaterEqual(pc.headroom_bytes, 0)
        print(
            f"\n[plan 32B ctx=2048] mac layers [0,38) used {mac.used_bytes / GiB:.3f} of 11.00 GiB; "
            f"pc layers [38,64) used {pc.used_bytes / GiB:.3f} of 6.50 GiB"
        )

    def test_real_budgets_refused_at_4096_context(self):
        with self.assertRaises(PlanError) as cm:
            plan_qwen3_32b(11.0, 6.5, context_tokens=4096)
        self.assertIn("short by", str(cm.exception))

    def test_cuda_reserve_refuses_2048_context(self):
        with self.assertRaises(PlanError):
            plan_qwen3_32b(11.0, 6.5, context_tokens=2048, pc_reserve_gib=0.5)

    def test_fp16_client_refused(self):
        with self.assertRaises(PlanError):
            plan_qwen3_32b(11.0, 6.5, context_tokens=2048, client="fp16")

    def test_weight_constants(self):
        self.assertEqual(MLX4_LAYER_BYTES, 274_289_152)
        self.assertEqual(NF4_LAYER_BYTES, 251_561_339)


class PlanRules(unittest.TestCase):
    def test_plan_that_does_not_fit_is_refused(self):
        mac = Member("mac", "metal", 8 * GiB, MLX4_LAYER_BYTES, fixed_bytes=875_161_600)
        pc = Member("pc", "cuda", int(6.5 * GiB), NF4_LAYER_BYTES)
        with self.assertRaises(PlanError) as cm:
            plan_split(64, mac, pc, kv_bytes_per_token_per_layer=KV, context_tokens=2048)
        self.assertIn("no split of 64 layers fits", str(cm.exception))

    def test_cpu_only_member_is_refused(self):
        with self.assertRaises(PlanError) as cm:
            Member("pc", "cpu", 6 * GiB, NF4_LAYER_BYTES)
        self.assertIn("CPU-only member is refused", str(cm.exception))

    def test_split_covers_every_layer_once(self):
        mac = Member("mac", "metal", 11 * GiB, MLX4_LAYER_BYTES, fixed_bytes=875_161_600)
        pc = Member("pc", "cuda", 6 * GiB, NF4_LAYER_BYTES)
        for ctx in (256, 512, 1024):
            plan = plan_split(64, mac, pc, kv_bytes_per_token_per_layer=KV, context_tokens=ctx)
            route = plan.route()
            self.assertEqual(route[0][1], 0)
            self.assertEqual(route[-1][2], 64)
            self.assertEqual(route[0][2], route[1][1])

    def test_bad_context_refused(self):
        mac = Member("mac", "metal", 11 * GiB, MLX4_LAYER_BYTES)
        pc = Member("pc", "cuda", 6 * GiB, NF4_LAYER_BYTES)
        with self.assertRaises(PlanError):
            plan_split(64, mac, pc, kv_bytes_per_token_per_layer=KV, context_tokens=0)


if __name__ == "__main__":
    unittest.main()
