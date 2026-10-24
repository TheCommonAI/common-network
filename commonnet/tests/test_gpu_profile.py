"""The GPU profiling script, dry-run on a tiny random Qwen3 on the CPU (no GPU needed)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from commonnet.bench import gpu_profile  # noqa: E402
from commonnet.tests.test_split import make_checkpoint  # noqa: E402


class GpuProfile(unittest.TestCase):
    def test_cpu_dry_run_writes_a_report_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, out = os.path.join(tmp, "tiny"), os.path.join(tmp, "report.json")
            make_checkpoint(ckpt)
            code = gpu_profile.main(["--layers", "0-2", "--backend", "cpu", "--checkpoint", ckpt, "--steps", "5",
                                     "--profile-steps", "2", "--prefill", "4", "--out", out])
            self.assertEqual(code, 0)
            report = json.load(open(out))
        self.assertEqual(report["info"]["layers"], [0, 2])
        self.assertGreater(report["info"]["weight_bytes"], 0)
        self.assertEqual(report["timing"]["steps"], 5)
        self.assertGreater(report["timing"]["wall_ms_median"], 0)
        self.assertIn("top_by_cpu", report["profile"])
        self.assertNotIn("token", json.dumps(report).lower())

    def test_layers_are_parsed(self):
        self.assertEqual(gpu_profile.parse_layers("27-36"), (27, 36))

    def test_verdict_names_launch_bound_when_the_gpu_is_mostly_idle(self):
        timing = {"wall_ms_median": 30.0, "python_launch_ms_median": 25.0}
        lines = gpu_profile.verdict(timing, 1 * 2**30, 270.0, gpu_kernel_ms=8.0)
        self.assertTrue(any("launch/Python bound" in l for l in lines))

    def test_patricks_rtx_4060_run_reads_as_launch_bound_not_memory_bound(self):
        # 0.81 GiB of weights, 234 GB/s measured, 18.8 ms step, 9.7 ms of GPU kernels (real run, 24 Oct 2026)
        timing = {"wall_ms_median": 18.8, "gpu_event_ms_median": 18.8, "python_launch_ms_median": 18.7}
        lines = gpu_profile.verdict(timing, int(0.81 * 2**30), 234.0, gpu_kernel_ms=9.7)
        self.assertTrue(any("floor for reading the weights once per token: 3.7 ms" in l for l in lines), lines)
        self.assertTrue(any("launch/Python bound" in l for l in lines), lines)

    def test_verdict_names_memory_bound_when_the_gpu_is_busy_near_the_floor(self):
        timing = {"wall_ms_median": 9.0, "python_launch_ms_median": 2.0}
        lines = gpu_profile.verdict(timing, 1 * 2**30, 270.0, gpu_kernel_ms=8.5)
        self.assertTrue(any("memory-bound" in l for l in lines))


if __name__ == "__main__":
    unittest.main()
