"""`common join --direct`: the flags reach the node, and nothing changes without them."""
import argparse
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import join  # noqa: E402

GLOBAL = "93.184.216.34"


class DirectFlags(unittest.TestCase):
    def test_without_direct_nothing_is_added(self):
        off = argparse.Namespace(network="", no_update=True, api=None, layers=None, direct=False, bootstrap=[])
        self.assertEqual(join._direct_flags(off), ["--auto-direct"])   # plain join tries direct, falls back to the tunnel
        self.assertEqual(join._direct_flags(argparse.Namespace(direct=False, no_direct=True)), [])
        self.assertEqual(join._service_argv(off)[-1], "--network")
        self.assertNotIn("--direct", join._service_argv(off))

    def test_service_argv_carries_every_direct_option(self):
        on = argparse.Namespace(network="", no_update=True, api=11440, layers=None, direct=True,
                                bootstrap=["1.2.3.4:5", "6.7.8.9:10"], public_ip=GLOBAL, external_port=7000,
                                dht_port=7001, direct_port=7000, no_tunnel_fallback=True)
        argv = join._service_argv(on)
        for flag in ("--direct", "--bootstrap=1.2.3.4:5", "--bootstrap=6.7.8.9:10", f"--public-ip={GLOBAL}",
                     "--external-port=7000", "--dht-port=7001", "--port=7000", "--no-tunnel-fallback"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[-1], "--network")    # the spec still goes last

    def test_direct_options_without_direct_are_a_parse_error(self):
        import subprocess
        out = subprocess.run([sys.executable, str(HERE / "join.py"), "--no-update", "--bootstrap", "1.2.3.4:5"],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 2)
        self.assertIn("need --direct", out.stderr)


if __name__ == "__main__":
    unittest.main()
