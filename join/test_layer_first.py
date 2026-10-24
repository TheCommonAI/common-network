"""`common join` defaults to the layer network; the whole-model gateway node is --legacy; services keep their mode."""
import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOIN_PY = HERE / "join.py"
sys.path.insert(0, str(HERE))
import join  # noqa: E402


def ns(**kw):
    base = dict(network=None, legacy=False, model=None, lan=False, list_catalogue=False, remove_permanent=False,
                permanent=False, no_update=True, gateway="https://g.example", name="n", operator="o", cost=0.0,
                region=None, capability=None)
    base.update(kw)
    return argparse.Namespace(**base)


class ChooseMode(unittest.TestCase):
    def test_a_plain_join_is_the_layer_network_when_the_machine_can_serve_it(self):
        self.assertEqual(join.choose_mode(ns(), supported=lambda: True), "network")

    def test_a_plain_join_on_a_machine_that_cannot_stays_legacy(self):
        self.assertEqual(join.choose_mode(ns(), supported=lambda: False), "legacy")

    def test_legacy_and_anything_gateway_specific_imply_legacy(self):
        for kw in ({"legacy": True}, {"model": "qwen-x"}, {"lan": True}, {"list_catalogue": True},
                   {"remove_permanent": True}):
            with self.subTest(kw=kw):
                self.assertEqual(join.choose_mode(ns(**kw), supported=lambda: True), "legacy")

    def test_network_is_network(self):
        self.assertEqual(join.choose_mode(ns(network=""), supported=lambda: False), "network")
        self.assertEqual(join.choose_mode(ns(network="https://x.example"), supported=lambda: True), "network")

    def test_a_service_installed_before_the_change_keeps_running_legacy(self):
        # Such a service always carried --model; it never saw --legacy.
        self.assertEqual(join.choose_mode(ns(model="qwen-x", permanent=False), supported=lambda: True), "legacy")


class ServiceKeepsItsMode(unittest.TestCase):
    def test_a_legacy_service_records_legacy_and_its_model(self):
        argv = join._service_argv(ns(model="qwen-x", lan=True, region="au"))
        self.assertIn("--legacy", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "qwen-x")
        self.assertNotIn("--network", " ".join(argv))

    def test_a_network_service_records_network_and_no_gateway_model(self):
        argv = join._service_argv(ns(network=""))
        self.assertEqual(argv[-1], "--network")
        for flag in ("--model", "--gateway", "--legacy"):
            self.assertNotIn(flag, argv)
        self.assertEqual(join._service_argv(ns(network="https://x.example", no_update=True))[-2:],
                         ["--no-update", "--network=https://x.example"])

    def test_the_recorded_command_line_parses_back_to_the_same_mode(self):
        self.assertEqual(join.choose_mode(ns(legacy=True, model="qwen-x"), supported=lambda: True), "legacy")
        self.assertEqual(join.choose_mode(ns(network=""), supported=lambda: False), "network")


class Notice(unittest.TestCase):
    def test_the_change_is_announced_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker, lines = Path(tmp) / "m" / "notice", []
            self.assertTrue(join.announce_layer_first(marker, lines.append))
            self.assertFalse(join.announce_layer_first(marker, lines.append))
            self.assertEqual(len(lines), 1)
            self.assertIn("--legacy", lines[0])


HARNESS = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, {here!r})
    import join
    calls = []
    join.layer_network_supported = lambda: {supported}
    join.announce_layer_first = lambda *a, **k: calls.append("notice")
    join.run_network_member = lambda network, *a: calls.append(("network", network))
    join.install_permanent = lambda args: calls.append(("service", args.network))
    class Legacy(Exception): pass
    def legacy(*a, **k): raise Legacy()
    join.read_identity = legacy
    sys.argv = ["join.py", "--no-update"] + {argv!r}
    try:
        join.main()
    except Legacy:
        calls.append("legacy")
    print(json.dumps(calls))
''')


def dispatch(argv, supported=True):
    out = subprocess.run([sys.executable, "-c", HARNESS.format(here=str(HERE), argv=argv, supported=supported)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


class Dispatch(unittest.TestCase):
    def test_plain_join_goes_to_the_layer_network_and_says_so_once(self):
        self.assertEqual(dispatch([]), ["notice", ["network", None]])

    def test_legacy_flag_goes_to_the_gateway_path(self):
        self.assertEqual(dispatch(["--legacy"]), ["legacy"])

    def test_model_implies_legacy(self):
        self.assertEqual(dispatch(["--model", "qwen-x"]), ["legacy"])

    def test_plain_join_on_an_unsupported_machine_explains_and_exits(self):
        # the legacy gateway is retired, so a machine with no GPU is told so rather than sent there
        out = subprocess.run([sys.executable, "-c", HARNESS.format(here=str(HERE), argv=[], supported=False)],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 1)
        self.assertIn("NVIDIA GPU", out.stderr)
        self.assertNotIn("Traceback", out.stderr)

    def test_explicit_network_has_no_notice(self):
        self.assertEqual(dispatch(["--network", "https://x.example"]), [["network", "https://x.example"]])

    def test_permanent_installs_a_service_in_the_mode_it_would_run(self):
        self.assertEqual(dispatch(["--permanent"]), ["notice", ["service", ""]])
        self.assertEqual(dispatch(["--permanent", "--legacy"]), ["legacy"])  # legacy path installs after resolving a model


class RetiredGateway(unittest.TestCase):
    def test_gateway_410_is_a_clear_message_not_a_traceback(self):
        import urllib.error
        from unittest import mock
        import join

        err = urllib.error.HTTPError("http://x/assign", 410, "Gone", {}, None)
        with mock.patch.object(join, "http_json", side_effect=err), self.assertRaises(SystemExit):
            join.gateway_json("POST", "http://x/assign", body={})


if __name__ == "__main__":
    unittest.main()
