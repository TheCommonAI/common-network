"""The v0.2 RPC-group prototype is gone: its flags say so clearly, and everything else about `join.py` is unchanged."""
import argparse
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOIN_PY = HERE / "join.py"
sys.path.insert(0, str(HERE))
import join  # noqa: E402

REMOVED = (["--distributed-config", "c.json"], ["--local-only"], ["--group-token", "cng1.x.y"],
           ["--relay", "https://x.example"], ["--port", "11436"])
KEPT_FLAGS = ("--gateway", "--model", "--network", "--auto", "--list-catalogue", "--name", "--operator", "--cost",
              "--region", "--capability", "--lan", "--permanent", "--remove-permanent", "--no-update")


def run(*args):
    return subprocess.run([sys.executable, str(JOIN_PY), "--no-update", *args], capture_output=True, text=True, timeout=60)


class RemovedFlags(unittest.TestCase):
    def test_each_removed_flag_says_it_was_removed_and_points_to_network(self):
        for flag in REMOVED:
            with self.subTest(flag=flag):
                out = run(*flag)
                self.assertEqual(out.returncode, 2)
                self.assertIn("removed:", out.stderr)
                self.assertIn(flag[0], out.stderr)
                self.assertIn("--network", out.stderr)
                self.assertNotIn("Traceback", out.stderr)

    def test_removed_flags_are_hidden_from_help_and_the_rest_is_still_there(self):
        helptext = run("--help").stdout
        for flag, _ in [(f[0], None) for f in REMOVED]:
            self.assertNotIn(flag, helptext)
        for flag in KEPT_FLAGS:
            self.assertIn(flag, helptext)

    def test_removed_code_is_gone(self):
        for name in ("run_local_coordinator", "run_group_member"):
            self.assertFalse(hasattr(join, name), name)
        self.assertNotIn("group.py", join.SIBLING_MODULES)
        self.assertNotIn("distributed", join.start_llama_server.__code__.co_varnames)
        for name in ("group.py", "group_config.py", "DISTRIBUTED.md"):
            self.assertFalse((HERE / name).exists(), name)


class ServiceCommandLine(unittest.TestCase):
    def test_a_permanent_service_command_line_is_the_same_plus_the_recorded_mode(self):
        args = argparse.Namespace(gateway="https://g.example", model="qwen-x", name="n1", operator="o", cost=0.0,
                                  region=None, capability=None, no_update=False, lan=False)
        argv = join._service_argv(args)
        self.assertEqual(argv[2:], ["--legacy", "--gateway", "https://g.example", "--model", "qwen-x", "--name", "n1",
                                    "--operator", "o", "--cost", "0.0"])
        args.lan, args.region = True, "au"
        self.assertEqual(join._service_argv(args)[-3:], ["--region", "au", "--lan"])


if __name__ == "__main__":
    unittest.main()
