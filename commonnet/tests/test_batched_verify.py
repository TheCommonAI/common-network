"""The batched verify bench: table parsing, KV arithmetic and the verdict, with no llama.cpp binary needed."""
from __future__ import annotations

import unittest

from commonnet.bench import batched_verify as bv

TABLE = """\
llama_batched_bench: n_kv_max = 4096, n_batch = 2048
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|     6 |      1 |    1 |      7 |    0.024 |   248.70 |    0.014 |    69.43 |    0.039 |   181.68 |
|     6 |      1 |    4 |     28 |    0.039 |   616.25 |    0.022 |   185.67 |    0.060 |   462.89 |
"""

LOG = """\
0.00.132.621 I print_info: n_layer               = 28
0.00.132.642 I print_info: n_head_kv             = 8
0.00.132.644 I print_info: n_embd_head_k         = 128
0.00.132.644 I print_info: n_embd_head_v         = 128
"""


class ParseTable(unittest.TestCase):
    def test_rows_are_read_with_user_count_and_pass_seconds(self):
        rows = bv.parse_rows(TABLE)
        self.assertEqual([r["users"] for r in rows], [1, 4])
        self.assertEqual(rows[1]["pp"], 6)
        self.assertAlmostEqual(rows[1]["t_pp_s"], 0.039)

    def test_header_and_separator_are_ignored(self):
        self.assertEqual(bv.parse_rows("|    PP |  TG |\n|-------|\n"), [])

    def test_model_shape_is_read_from_the_log(self):
        self.assertEqual(bv.parse_model_info(LOG),
                         {"n_layer": 28, "n_head_kv": 8, "n_embd_head_k": 128})

    def test_missing_model_shape_is_an_error(self):
        with self.assertRaises(ValueError):
            bv.parse_model_info("print_info: n_layer = 28\n")


class KvAndVerdict(unittest.TestCase):
    def test_kv_per_user_is_split_across_stages(self):
        info = {"n_layer": 28, "n_head_kv": 8, "n_embd_head_k": 128}
        one = bv.kv_report(info, ctx_per_seq=1024, stages=1, bytes_per_elem=2)
        three = bv.kv_report(info, ctx_per_seq=1024, stages=3, bytes_per_elem=2)
        # 2 (K,V) * 8 heads * 128 dims * 2 bytes * 28 layers * 1024 cells = 114688 KiB = 112 MiB
        self.assertAlmostEqual(one["per_user_total_mib"], 112.0)
        self.assertAlmostEqual(three["per_user_per_stage_mib"], 112.0 / 3, places=1)

    def _table(self, ms_by_users):
        return [{"users": u, "verify_ms_median": ms} for u, ms in ms_by_users.items()]

    def test_flat_cost_is_called_batching_holds(self):
        self.assertIn("about the same", bv.verdict(self._table({1: 40.0, 4: 44.0})))

    def test_linear_cost_is_called_plan_copies(self):
        self.assertIn("multiple model copies", bv.verdict(self._table({1: 40.0, 4: 150.0})))

    def test_in_between_is_reported_as_partly_batched(self):
        self.assertIn("partly batched", bv.verdict(self._table({1: 40.0, 4: 80.0})))

    def test_the_verdict_is_judged_at_four_users_when_measured(self):
        self.assertIn("about the same", bv.verdict(self._table({1: 40.0, 4: 44.0, 8: 400.0})))

    def test_needs_a_one_user_baseline(self):
        self.assertIn("need", bv.verdict(self._table({4: 80.0})))


if __name__ == "__main__":
    unittest.main()
