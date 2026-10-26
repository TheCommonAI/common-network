"""Whole-network layer assignment (commonnet.mesh.assign). Pure functions over signed records; no model, no network."""
import itertools
import json
import random
import unittest
from unittest import mock

from commonnet import coordinator
from commonnet.mesh import assign
from commonnet.mesh.assign import AssignmentError, changes, check_cover, plan
from commonnet.mesh.manifest import Manifest
from commonnet.mesh.records import Identity

NOW = 10_000.0
MANIFEST = Manifest.from_dict({
    "id": "t", "model": {"name": "T", "num_layers": 36, "repos": {"mlx": "a/b"}},
    "boundaries": [0, 9, 18, 27, 36], "gib_per_layer": {"mlx": 0.125, "nf4": 0.125}, "fixed_gib": 0.3})
# mlx: one stage 1.425 GiB, (0,18) and (18,36) 2.55, (0,27) 3.675, the whole model 4.8


class Node:
    """A peer that can be re-signed on another window: same identity, so the same peer id throughout."""

    def __init__(self, gib=2.6, window=None, speed=0.0, backend="mlx"):
        self.who = Identity.generate()
        self.gib, self.speed, self.backend = gib, speed, backend
        self.window = window

    @property
    def peer(self):
        return self.who.peer

    def record(self, window="keep"):
        if window != "keep":
            self.window = window
        fields = {"network": "t", "url": "http://127.0.0.1:1", "token": "t", "window": self.window,
                  "backend": self.backend, "state": "up", "gib": self.gib}
        if self.speed:
            fields["speed"] = self.speed
        return self.who.sign_record(fields, now=NOW)


def fits(record, window, manifest=MANIFEST):
    return manifest.window_gib(*window, record["backend"]) <= record["gib"]


def placed(nodes, result):
    """Records of `nodes` re-signed on the windows `result` gives them (idle peers keep no window)."""
    return [n.record(result.windows.get(n.peer)) for n in nodes]


def windows_now(records):
    return {r["peer"]: tuple(r["window"]) if r.get("window") else None for r in records}


class Fresh(unittest.TestCase):
    def test_two_peers_that_each_hold_half_make_the_chain(self):
        a, b = Node(), Node()
        result = plan(MANIFEST, [a.record(), b.record()])
        self.assertEqual(result.chain, ((0, 18), (18, 36)))
        self.assertEqual(sorted(result.windows.values()), [(0, 18), (18, 36)])
        self.assertEqual(result.idle, ())

    def test_one_peer_that_holds_everything_is_one_hop(self):
        big, small = Node(gib=5.0), Node(gib=2.6)
        result = plan(MANIFEST, [big.record(), small.record()])
        self.assertEqual(result.chain, ((0, 36),))
        self.assertEqual(result.windows[big.peer], (0, 36))
        self.assertIn(small.peer, result.windows)  # a replica, not idle

    def test_a_peer_with_too_little_memory_is_idle(self):
        tiny, a, b = Node(gib=1.0), Node(), Node()
        result = plan(MANIFEST, [tiny.record(), a.record(), b.record()])
        self.assertEqual(result.idle, (tiny.peer,))
        self.assertNotIn(tiny.peer, result.windows)

    def test_a_peer_that_is_not_up_is_ignored(self):
        loading = {**Node().record(), "state": "loading"}
        a, b = Node(), Node()
        result = plan(MANIFEST, [loading, a.record(), b.record()])
        self.assertNotIn(loading["peer"], result.windows)
        self.assertEqual(len(result.chain), 2)

    def test_faster_peers_take_the_chain_and_slower_ones_become_replicas(self):
        fast, medium, slow = Node(speed=100.0), Node(speed=50.0), Node(speed=1.0)
        result = plan(MANIFEST, [slow.record(), fast.record(), medium.record()])
        self.assertEqual(result.chain, ((0, 18), (18, 36)))
        self.assertEqual({result.windows[fast.peer], result.windows[medium.peer]}, {(0, 18), (18, 36)})
        self.assertIn(slow.peer, result.windows)

    def test_no_peers_is_a_clear_error(self):
        with self.assertRaises(AssignmentError) as caught:
            plan(MANIFEST, [])
        self.assertIn("no live peers", str(caught.exception))

    def test_nobody_can_hold_a_stage_names_the_layers(self):
        with self.assertRaises(AssignmentError) as caught:
            plan(MANIFEST, [Node(gib=1.0).record()])
        self.assertIn("layers 0-36", str(caught.exception))
        self.assertIn("memory", str(caught.exception))

    def test_a_given_previous_plan_is_respected(self):
        a, b = Node(), Node()
        first = plan(MANIFEST, [a.record(), b.record()])
        swapped = {a.peer: first.windows[b.peer], b.peer: first.windows[a.peer]}
        again = plan(MANIFEST, [a.record(), b.record()], previous=swapped)
        self.assertEqual(again.windows, swapped)


class Join(unittest.TestCase):
    """A joining peer changes nobody else's window, so no other node restarts."""

    def setUp(self):
        self.a, self.b = Node(), Node()
        first = plan(MANIFEST, [self.a.record(), self.b.record()])
        self.live = placed([self.a, self.b], first)

    def test_a_replica_joins_and_nobody_moves(self):
        c = Node()
        before = windows_now(self.live)
        result = plan(MANIFEST, self.live + [c.record()])
        self.assertEqual(changes(before, result), {})
        self.assertEqual(result.chain, ((0, 18), (18, 36)))
        self.assertTrue(fits(c.record(), result.windows[c.peer]))

    def test_a_larger_peer_that_can_hold_everything_gets_one_hop_and_nobody_moves(self):
        big = Node(gib=5.0)
        before = windows_now(self.live)
        result = plan(MANIFEST, self.live + [big.record()])
        self.assertEqual(result.chain, ((0, 36),))
        self.assertEqual(changes(before, result), {})

    def test_joining_repeatedly_moves_nobody(self):
        nodes = [self.a, self.b]
        windows = {self.a.peer: (0, 18), self.b.peer: (18, 36)}
        for _ in range(4):
            nodes.append(Node())
            result = plan(MANIFEST, [n.record(windows.get(n.peer)) for n in nodes], previous=windows)
            self.assertEqual(changes(windows, result), {})
            windows = dict(result.windows)


class Drop(unittest.TestCase):
    def test_losing_a_replica_moves_nobody(self):
        a, b, spare = Node(window=(0, 18)), Node(window=(18, 36)), Node(window=(18, 36))
        before = {a.peer: (0, 18), b.peer: (18, 36), spare.peer: (18, 36)}
        result = plan(MANIFEST, [a.record(), spare.record()], previous=before)
        self.assertEqual(result.chain, ((0, 18), (18, 36)))
        self.assertEqual(changes(before, result).keys(), {b.peer})  # only the dropped peer is gone, nobody moved
        self.assertEqual(changes(before, result)[b.peer], ((18, 36), None))

    def test_losing_a_chain_holder_is_covered_by_a_spare_and_only_one_peer_moves(self):
        a, spare = Node(window=(0, 18)), Node(window=(0, 18))
        before = {a.peer: (0, 18), spare.peer: (0, 18)}  # the (18, 36) holder is gone
        result = plan(MANIFEST, [a.record(), spare.record()], previous=before)
        self.assertEqual(result.chain, ((0, 18), (18, 36)))
        self.assertEqual(sorted(result.windows.values()), [(0, 18), (18, 36)])
        self.assertEqual(len(changes(before, result)), 1)

    def test_losing_the_only_holder_of_a_stage_fails_with_the_layers_named(self):
        a = Node(window=(0, 18))
        with self.assertRaises(AssignmentError) as caught:
            plan(MANIFEST, [a.record()], previous={a.peer: (0, 18)})
        message = str(caught.exception)
        self.assertIn("layers 0-18", message)
        self.assertIn("layers 18-36", message)


class Speed(unittest.TestCase):
    def test_speed_is_read_from_signed_records_and_decides_who_serves_the_chain(self):
        slow, fast, unknown = Node(speed=1.0), Node(speed=80.0), Node(speed=0.0)
        result = plan(MANIFEST, [slow.record(), fast.record(), unknown.record()])
        self.assertEqual(len(result.chain), 2)
        self.assertIn(fast.peer, result.windows)
        self.assertIn(fast.peer, [p for p, w in result.windows.items() if w in result.chain])


class ChainInvariants(unittest.TestCase):
    """Whatever the peers are, a returned chain tiles every layer, and a window goes to a peer that can hold it."""

    def test_random_networks_give_valid_tilings_and_an_error_only_when_no_tiling_is_feasible(self):
        rng = random.Random(7)
        for trial in range(300):
            nodes = [Node(gib=round(rng.uniform(0.5, 5.5), 2), speed=rng.choice([0.0, 1.0, 10.0, 50.0]),
                          backend=rng.choice(["mlx", "nf4"]))
                     for _ in range(rng.randint(1, 5))]
            records = [n.record() for n in nodes]
            with self.subTest(trial=trial):
                feasible = _some_tiling_is_servable(MANIFEST, records)
                try:
                    result = plan(MANIFEST, records)
                except AssignmentError:
                    self.assertFalse(feasible, "the planner gave up but a covering assignment exists")
                    continue
                self.assertTrue(feasible)
                check_cover(MANIFEST, result.chain)
                by_id = {r["peer"]: r for r in records}
                served = set(result.windows.values())
                for window in result.chain:
                    self.assertIn(window, served)
                for pid, window in result.windows.items():
                    self.assertTrue(fits(by_id[pid], window), (pid, window))
                self.assertEqual(set(result.windows) | set(result.idle), set(by_id))

    def test_check_cover_rejects_gaps_overlaps_and_off_boundary_windows(self):
        with self.assertRaises(AssignmentError) as caught:
            check_cover(MANIFEST, [(0, 9), (18, 36)])
        self.assertIn("9-18", str(caught.exception))
        with self.assertRaises(AssignmentError) as caught:
            check_cover(MANIFEST, [(0, 18), (9, 36)])
        self.assertIn("assigned twice", str(caught.exception))
        with self.assertRaises(AssignmentError):
            check_cover(MANIFEST, [(0, 18)])
        with self.assertRaises(AssignmentError):
            check_cover(MANIFEST, [(0, 10), (10, 36)])
        check_cover(MANIFEST, [(0, 9), (9, 27), (27, 36)])


def _some_tiling_is_servable(manifest, records) -> bool:
    """Brute force: some tiling of the stages gives each of its windows a different peer that fits it."""
    for tiling in assign._tilings(manifest.boundaries):
        for chosen in itertools.permutations(range(len(records)), len(tiling)):
            if all(fits(records[i], w, manifest) for i, w in zip(chosen, tiling)):
                return True
    return False


class Output(unittest.TestCase):
    def test_to_dict_is_json_serialisable(self):
        result = plan(MANIFEST, [Node().record(), Node().record(), Node(gib=1.0).record()])
        data = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(data["chain"], [[0, 18], [18, 36]])
        self.assertEqual(len(data["idle"]), 1)


class PrintPlan(unittest.TestCase):
    def test_prints_the_plan_of_a_live_network(self):
        network = mock.Mock()
        network.manifest = MANIFEST
        network.table.live.return_value = [Node().record(), Node().record()]
        with mock.patch("commonnet.netclient.Network.from_spec", return_value=network), \
                mock.patch("builtins.print") as printed:
            self.assertEqual(coordinator.print_plan("https://peer.example"), 0)
        payload = json.loads(printed.call_args[0][0])
        self.assertEqual(payload["chain"], [[0, 18], [18, 36]])
        network.refresh.assert_called_once_with(force=True)

    def test_an_uncovered_network_exits_1_with_the_reason(self):
        network = mock.Mock()
        network.manifest = MANIFEST
        network.table.live.return_value = [Node(gib=1.0).record()]
        with mock.patch("commonnet.netclient.Network.from_spec", return_value=network), \
                mock.patch("sys.stderr") as err:
            self.assertEqual(coordinator.print_plan("https://peer.example"), 1)
        self.assertIn("layers 0-36", "".join(c.args[0] for c in err.write.call_args_list))

    def test_main_needs_a_prompt_unless_it_is_planning(self):
        self.assertEqual(coordinator.main(["--network", "https://peer.example"]), 2)


if __name__ == "__main__":
    unittest.main()
