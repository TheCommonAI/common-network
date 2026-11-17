"""The public join CLI chooses background service operation without changing internal launchers."""
import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


class BackgroundJoin(unittest.TestCase):
    def command(self, rest=(), *, verb="join", model=None):
        args = argparse.Namespace(no_update=True, auto=False, model=model, region=None, rest=list(rest))
        with mock.patch.object(common, "_exec") as execute:
            common.cmd_join_or_serve(verb, "https://gateway.example", args)
        return execute.call_args.args[0]

    def test_plain_join_installs_the_background_service(self):
        argv = self.command()
        self.assertEqual(argv.count("--permanent"), 1)
        self.assertNotIn("--foreground", argv)

    def test_foreground_leaves_terminal_operation_explicit(self):
        argv = self.command(["--foreground"])
        self.assertIn("--foreground", argv)
        self.assertNotIn("--permanent", argv)

    def test_explicit_permanent_is_forwarded_once(self):
        self.assertEqual(self.command(["--permanent"]).count("--permanent"), 1)

    def test_action_only_commands_never_install_a_background_node(self):
        for flag in ("--check", "--controls", "--remove-permanent", "--help", "-h"):
            with self.subTest(flag=flag):
                argv = self.command([flag])
                self.assertIn(flag, argv)
                self.assertNotIn("--permanent", argv)

    def test_no_controls_and_resource_choices_are_kept_with_background_default(self):
        flags = ["--no-controls", "--memory-gib=4", "--cpu-threads=2", "--compute-percent=35", "--bandwidth-mbps=10"]
        argv = self.command(flags)
        self.assertIn("--permanent", argv)
        self.assertEqual(argv[-len(flags):], flags)

    def test_presets_and_named_networks_use_the_background_default(self):
        for flags in (["--7"], ["--nodes", "2"], ["--network", "qwen3-8b"]):
            with self.subTest(flags=flags):
                argv = self.command(flags)
                self.assertIn("--permanent", argv)
                self.assertEqual(argv[-len(flags):], flags)

    def test_serve_and_retired_explicit_modes_keep_their_existing_behavior(self):
        self.assertNotIn("--permanent", self.command(verb="serve"))
        for flag in ("--legacy", "--lan", "--list-catalogue"):
            with self.subTest(flag=flag):
                self.assertNotIn("--permanent", self.command([flag]))


if __name__ == "__main__":
    unittest.main()
