"""Preset selection, gap filling, replica placement and discovery over real local HTTP."""
import unittest

from commonnet import node
from commonnet.mesh import placement
from commonnet.mesh.records import Identity
from commonnet.presets import model_for_nodes, placement_manifest


def manifest(count):
    selected, _ = node.resolve_network(model_for_nodes(count), [], environ={})
    return placement_manifest(selected, count)


class Presets(unittest.TestCase):
    def test_models_and_shared_network_ids(self):
        self.assertEqual([model_for_nodes(n) for n in range(1, 8)],
                         ["qwen3-8b"] * 2 + ["qwen3.5-35b-a3b"] * 4 + ["qwen3.5-122b-a10b"])
        self.assertEqual(manifest(1).id, manifest(2).id)
        self.assertEqual(len({manifest(n).id for n in range(3, 7)}), 1)
        self.assertNotEqual(manifest(3).id, manifest(7).id)
        for bad in (0, 8, -1, True, "7"):
            with self.assertRaises(ValueError):
                model_for_nodes(bad)

    def test_sequential_mixed_gpu_peers_cover_model_before_adding_replicas(self):
        for count, complete_after in ((1, 1), (2, 2), (3, 3), (4, 3), (5, 3), (6, 3), (7, 7)):
            with self.subTest(count=count):
                network, live = manifest(count), []
                for i in range(count):
                    backend = "mlx" if i % 2 == 0 else "nf4"
                    w = placement.choose_window(network, live, 64, backend)
                    if i >= complete_after:
                        self.assertIn(w, [tuple(r["window"]) for r in live])
                    live.append({"peer": str(i), "window": list(w), "state": "up", "backend": backend,
                                 "url": f"https://node-{i}.example", "gib": 64})
                route = placement.best_route(network, live)
                self.assertEqual(route[0].window[0], 0)
                self.assertEqual(route[-1].window[1], network.num_layers)
                self.assertLessEqual(max(b - a for a, b in (r.window for r in route)), network.model["max_window_layers"])
                if count == 6:
                    self.assertTrue(all(value >= 2 for value in placement.coverage(network, live)))

    def test_seven_node_plan_does_not_claim_a_complete_route_with_two_peers(self):
        network = manifest(7)
        live = [{"peer": "a", "state": "up", "window": [0, 7]},
                {"peer": "b", "state": "up", "window": [7, 14]}]
        with self.assertRaises(placement.NoRoute):
            placement.best_route(network, live)
        self.assertEqual(placement.choose_window(network, live, 16, "mlx"), (14, 21))

    def test_small_machine_is_rejected_before_downloading(self):
        with self.assertRaises(placement.NoWindow):
            placement.choose_window(manifest(7), [], 2, "nf4")

    def test_gap_fill_bridges_existing_prefix_and_suffix_without_unusable_overlap(self):
        network = manifest(3)
        live = [{"peer": "a", "state": "up", "window": [0, 16]},
                {"peer": "b", "state": "up", "window": [24, 40]}]
        self.assertEqual(placement.choose_window(network, live, 64, "nf4"), (16, 24))

    def test_simultaneous_duplicate_placements_can_move_to_fill_missing_stages(self):
        network = manifest(3)
        live = [{"peer": name, "state": "up", "window": [0, 16], "gib": 64, "backend": "mlx"}
                for name in ("a", "b", "c")]
        moved = placement.should_move(network, live, "a")
        self.assertEqual(moved, (16, 32))
        live[0]["window"] = list(moved)
        moved = placement.should_move(network, live, "b")
        self.assertEqual(moved, (32, 40))
        live[1]["window"] = list(moved)
        self.assertEqual(len(placement.best_route(network, live)), 3)

    def test_new_node_discovers_existing_slices_and_fills_the_gap(self):
        network, peers, logs = manifest(7), [], []
        try:
            for _ in range(3):
                peer = node.Node(network, [peers[0].url] if peers else [], Identity.generate(),
                                 node.Hardware("stub", 64, "stub"), stub=True, tunnel=False,
                                 resign_s=60, gossip_s=60, log=logs.append)
                peers.append(peer)
                peer.start()
            self.assertEqual([p.window for p in peers], [(0, 7), (7, 14), (14, 21)])
            self.assertTrue(any("waiting for more peers" in line for line in logs))
        finally:
            for peer in reversed(peers):
                peer.stop()


if __name__ == "__main__":
    unittest.main()
