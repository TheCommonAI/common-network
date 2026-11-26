"""Automatic Bonsai fallback when the requested layer network has no complete route."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from commonnet.coordinator import FallbackUnavailable, select_network_with_bonsai_fallback
from commonnet.mesh.assign import AssignmentError
from commonnet.mesh.manifest import Manifest


def manifest(network_id, name, gib_per_layer):
    return Manifest.from_dict({
        "id": network_id,
        "model": {"name": name, "num_layers": 2, "repos": {"cpu": "test/model"}},
        "boundaries": [0, 2],
        "gib_per_layer": {"cpu": gib_per_layer},
    })


class FakeNetwork:
    def __init__(self, value):
        self.manifest = value
        self.table = SimpleNamespace(live=lambda: [{"network": value.id}])
        self.refresh_calls = 0

    def refresh(self, force=False):
        self.refresh_calls += 1


class NetworkFallback(unittest.TestCase):
    def setUp(self):
        self.hardware = SimpleNamespace(backend="cpu", gib=2.0, name="test CPU")
        self.primary = FakeNetwork(manifest("full-model", "Full model", 3.0))

    def test_available_full_model_does_not_consider_fallbacks(self):
        candidate_source = lambda: self.fail("fallback catalogue should not be loaded")
        hardware_source = lambda: self.fail("hardware should not be redetected")

        selected, used_fallback = select_network_with_bonsai_fallback(
            self.primary, candidate_source, hardware_source, coverage_check=lambda _: None)

        self.assertIs(selected, self.primary)
        self.assertFalse(used_fallback)
        self.assertEqual(self.primary.refresh_calls, 1)

    def test_insufficient_coverage_uses_largest_locally_fitting_bonsai_with_coverage(self):
        large = FakeNetwork(manifest("bonsai-27", "Ternary Bonsai 27B", 0.75))
        small = FakeNetwork(manifest("bonsai-8", "Ternary Bonsai 8B", 0.25))
        checked = []

        def check(network):
            checked.append(network.manifest.id)
            if network is self.primary or network is large:
                raise AssignmentError("not enough peers")

        selected, used_fallback = select_network_with_bonsai_fallback(
            self.primary, [small, large], self.hardware, coverage_check=check)

        self.assertIs(selected, small)
        self.assertTrue(used_fallback)
        self.assertEqual(checked, ["full-model", "bonsai-27", "bonsai-8"])

    def test_no_locally_fitting_bonsai_returns_a_clear_error(self):
        too_large = FakeNetwork(manifest("bonsai-27", "Ternary Bonsai 27B", 2.0))

        with self.assertRaises(FallbackUnavailable) as caught:
            select_network_with_bonsai_fallback(
                self.primary, [too_large], self.hardware,
                coverage_check=lambda _: (_ for _ in ()).throw(AssignmentError("not enough peers")))

        self.assertIn("no Bonsai model/window fits test CPU", str(caught.exception))

    def test_records_from_a_different_manifest_are_never_reused(self):
        mismatched = FakeNetwork(manifest("bonsai-8", "Ternary Bonsai 8B", 0.25))
        mismatched.table = SimpleNamespace(live=lambda: [{"network": "another-model"}])
        checked = []

        def check(network):
            checked.append(network.manifest.id)
            if network is self.primary:
                raise AssignmentError("not enough peers")

        with self.assertRaises(FallbackUnavailable):
            select_network_with_bonsai_fallback(
                self.primary, [mismatched], self.hardware,
                coverage_check=check)

        self.assertEqual(checked, ["full-model"])


if __name__ == "__main__":
    unittest.main()
