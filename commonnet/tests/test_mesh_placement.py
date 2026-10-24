import unittest

from commonnet.mesh import placement
from commonnet.mesh.manifest import Manifest
from commonnet.mesh.placement import NoRoute, NoWindow, best_route, choose_window, coverage, should_move
from commonnet.mesh.records import Identity

NOW = 10_000.0
MANIFEST = Manifest.from_dict({
    "id": "t", "model": {"name": "T", "num_layers": 36, "repos": {"mlx": "a/b"}},
    "boundaries": [0, 9, 18, 27, 36], "gib_per_layer": {"mlx": 0.125, "nf4": 0.125}, "fixed_gib": 0.3})
# one stage = 1.425 GiB, two = 2.55, three = 3.675, four = 4.8


def peer(window, backend="mlx", gib=8.0, state="up", ts=NOW, who=None):
    who = who or Identity.generate()
    return who.sign_record({"network": "t", "url": "http://127.0.0.1:1", "token": "t", "window": window,
                            "backend": backend, "state": state, "gib": gib}, now=ts)


class Coverage(unittest.TestCase):
    def test_counts_peers_per_stage(self):
        live = [peer([0, 18]), peer([0, 9]), peer([27, 36])]
        self.assertEqual(coverage(MANIFEST, live), [2, 1, 0, 1])

    def test_loading_and_unwindowed_peers_do_not_count(self):
        live = [peer(None, state="loading"), peer([0, 9], state="loading")]
        self.assertEqual(coverage(MANIFEST, live), [0, 0, 0, 0])

    def test_a_window_off_the_boundaries_is_ignored(self):
        self.assertEqual(coverage(MANIFEST, [peer([0, 10])]), [0, 0, 0, 0])


class ChooseWindow(unittest.TestCase):
    def test_an_empty_network_gets_the_longest_window_that_fits(self):
        self.assertEqual(choose_window(MANIFEST, [], 2.6, "mlx"), (0, 18))
        self.assertEqual(choose_window(MANIFEST, [], 100, "mlx"), (0, 36))

    def test_uncovered_stages_come_first(self):
        live = [peer([0, 9]), peer([9, 18]), peer([27, 36])]
        self.assertEqual(choose_window(MANIFEST, live, 1.5, "mlx"), (18, 27))

    def test_an_uncovered_stage_beats_a_longer_window_over_covered_ones(self):
        live = [peer([9, 36]), peer([9, 36]), peer([9, 36])]
        self.assertEqual(choose_window(MANIFEST, live, 2.6, "mlx"), (0, 18))

    def test_a_tie_goes_to_a_window_that_already_exists_so_failover_has_a_twin(self):
        live = [peer([0, 18]), peer([18, 36])]
        for _ in range(5):
            self.assertIn(choose_window(MANIFEST, live, 2.6, "mlx"), [(0, 18), (18, 36)])

    def test_prefers_the_existing_window_with_fewer_replicas(self):
        live = [peer([0, 9]), peer([0, 9]), peer([9, 18]), peer([18, 27]), peer([27, 36])]
        # stage 0 has two peers, the others one each: the new peer helps most where there is one
        self.assertNotEqual(choose_window(MANIFEST, live, 1.5, "mlx"), (0, 9))

    def test_the_window_fits_the_memory_offered(self):
        for gib in (1.5, 2.6, 3.7, 4.9):
            w = choose_window(MANIFEST, [], gib, "mlx")
            self.assertLessEqual(MANIFEST.window_gib(*w, "mlx"), gib)

    def test_too_little_memory_for_one_stage(self):
        with self.assertRaises(NoWindow):
            choose_window(MANIFEST, [], 1.0, "mlx")

    def test_my_own_record_does_not_count_against_me(self):
        me = peer([0, 9])
        self.assertEqual(choose_window(MANIFEST, [me], 1.5, "mlx", me=me["peer"]), (0, 9))


class ShouldMove(unittest.TestCase):
    def setUp(self):
        self.me = peer([0, 18], gib=3.0)
        self.twin = peer([0, 18])
        self.mid = peer([18, 27])  # stage 3 (27-36) has nobody

    def test_moves_to_the_uncovered_stage_when_my_stages_have_twins(self):
        live = [self.me, self.twin, self.mid]
        new = should_move(MANIFEST, live, self.me["peer"])
        self.assertIsNotNone(new)
        self.assertGreaterEqual(new[1], 36)
        self.assertEqual(new[1], 36)

    def test_stays_when_everything_is_covered(self):
        live = [self.me, self.twin, self.mid, peer([27, 36])]
        self.assertIsNone(should_move(MANIFEST, live, self.me["peer"]))

    def test_stays_when_leaving_would_uncover_one_of_my_stages(self):
        live = [self.me, self.mid]  # nobody twins my window, stage 3 uncovered
        self.assertIsNone(should_move(MANIFEST, live, self.me["peer"]))

    def test_stays_when_it_cannot_hold_the_uncovered_window(self):
        small = peer([0, 18], gib=1.0)
        live = [small, self.twin, self.mid]
        self.assertIsNone(should_move(MANIFEST, live, small["peer"]))

    def test_unknown_or_loading_peer_does_not_move(self):
        live = [self.twin, self.mid]
        self.assertIsNone(should_move(MANIFEST, live, "nobody"))
        loading = peer(None, state="loading")
        self.assertIsNone(should_move(MANIFEST, live + [loading], loading["peer"]))


class Route(unittest.TestCase):
    def stage_peers(self, replicas=1, **kw):
        return [peer(list(w), **kw) for w in MANIFEST.stages() for _ in range(replicas)]

    def test_fewest_hops_wins(self):
        live = self.stage_peers() + [peer([0, 36])]
        route = best_route(MANIFEST, live, now=NOW)
        self.assertEqual([h.window for h in route], [(0, 36)])

    def test_a_two_stage_peer_beats_two_one_stage_peers(self):
        live = self.stage_peers() + [peer([9, 27])]
        route = best_route(MANIFEST, live, now=NOW)
        self.assertEqual([h.window for h in route], [(0, 9), (9, 27), (27, 36)])

    def test_alternates_are_identical_window_peers_same_backend_first(self):
        primary = peer([0, 36], "mlx", ts=NOW)
        older = peer([0, 36], "mlx", ts=NOW - 30)
        other_backend = peer([0, 36], "nf4", ts=NOW)
        different_window = peer([0, 18])
        route = best_route(MANIFEST, [other_backend, older, primary, different_window], prefer_backend="mlx", now=NOW)
        hop = route[0]
        self.assertEqual(hop.primary["peer"], primary["peer"])
        self.assertEqual([r["peer"] for r in hop.alternates], [older["peer"], other_backend["peer"]])

    def test_freshest_peer_is_primary(self):
        stale, fresh = peer([0, 36], ts=NOW - 50), peer([0, 36], ts=NOW)
        self.assertEqual(best_route(MANIFEST, [stale, fresh], now=NOW)[0].primary["peer"], fresh["peer"])

    def test_keeps_the_backend_of_the_previous_hop_when_it_costs_nothing(self):
        live = [peer([0, 18], "mlx"), peer([18, 36], "nf4"), peer([18, 36], "mlx")]
        route = best_route(MANIFEST, live, now=NOW)
        self.assertEqual([h.primary["backend"] for h in route], ["mlx", "mlx"])

    def test_prefer_backend(self):
        live = [peer([0, 36], "mlx"), peer([0, 36], "nf4")]
        self.assertEqual(best_route(MANIFEST, live, prefer_backend="nf4", now=NOW)[0].primary["backend"], "nf4")

    def test_no_route_names_the_boundary_where_it_stops(self):
        live = [peer([0, 9]), peer([18, 36])]
        with self.assertRaises(NoRoute) as caught:
            best_route(MANIFEST, live, now=NOW)
        self.assertEqual(caught.exception.boundary, 9)
        self.assertIn("9", str(caught.exception))

    def test_no_peers_at_all(self):
        with self.assertRaises(NoRoute) as caught:
            best_route(MANIFEST, [], now=NOW)
        self.assertEqual(caught.exception.boundary, 0)

    def test_loading_peers_are_not_routed_through(self):
        live = self.stage_peers()
        live[1] = {**live[1], "state": "loading"}
        with self.assertRaises(NoRoute):
            best_route(MANIFEST, live, now=NOW)

    def test_four_stages_two_replicas_survive_losing_any_one_peer(self):
        live = self.stage_peers(replicas=2)
        route = best_route(MANIFEST, live, now=NOW)
        self.assertEqual(len(route), 4)
        self.assertTrue(all(len(h.alternates) == 1 for h in route))
        for victim in live:
            with self.subTest(victim=victim["window"]):
                rest = [r for r in live if r["peer"] != victim["peer"]]
                again = best_route(MANIFEST, rest, now=NOW)
                self.assertEqual([h.window for h in again], MANIFEST.stages())

    def test_losing_both_replicas_of_a_stage_breaks_the_route(self):
        live = self.stage_peers(replicas=2)
        rest = [r for r in live if tuple(r["window"]) != (18, 27)]
        with self.assertRaises(NoRoute) as caught:
            best_route(MANIFEST, rest, now=NOW)
        self.assertEqual(caught.exception.boundary, 18)


if __name__ == "__main__":
    unittest.main()


class CpuPeers(unittest.TestCase):
    def test_a_gpu_peer_is_preferred_to_a_cpu_peer_for_the_same_window(self):
        cpu, gpu = peer([0, 9], backend="cpu"), peer([0, 9], backend="nf4")
        rest = [peer([9, 36], backend="nf4")]
        for order in ([cpu, gpu], [gpu, cpu]):
            route = best_route(MANIFEST, order + rest, now=NOW)
            self.assertEqual(route[0].primary["peer"], gpu["peer"])
            self.assertEqual(route[0].alternates[0]["peer"], cpu["peer"])

    def test_a_cpu_peer_still_serves_when_it_is_the_only_one(self):
        cpu = peer([0, 36], backend="cpu")
        self.assertEqual(best_route(MANIFEST, [cpu], now=NOW)[0].primary["peer"], cpu["peer"])

    def test_cpu_does_not_cost_extra_hops(self):
        # fewest hops still wins: one CPU peer for everything beats two GPU peers
        cpu, a, b = peer([0, 36], backend="cpu"), peer([0, 18], backend="nf4"), peer([18, 36], backend="nf4")
        self.assertEqual(len(best_route(MANIFEST, [cpu, a, b], now=NOW)), 1)
