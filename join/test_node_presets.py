"""The installed stdlib launcher accepts node-count shortcuts before loading a model runtime."""
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import join


class PresetFlags(unittest.TestCase):
    def test_shortcuts_and_long_flag_forward_the_count(self):
        for flags, count in [([f"--{n}"], n) for n in range(1, 8)] + [(["--nodes", "7"], 7)]:
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", ["join.py", "--no-update", *flags]), \
                    mock.patch.object(join, "run_network_member") as run, mock.patch("builtins.print"):
                join.main()
                run.assert_called_once_with(None, ["--api", "11440", "--nodes", str(count), "--auto-direct"], debug=False)

    def test_conflicting_and_out_of_range_counts_are_refused(self):
        for flags in (["--7", "--network", "qwen3-8b"], ["--1", "--2"], ["--nodes", "8"],
                      ["--7", "--model", "bonsai-8b"], ["--7", "--legacy"]):
            result = subprocess.run([sys.executable, str(Path(join.__file__)), "--no-update", *flags],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_permanent_service_preserves_count_without_a_conflicting_network_flag(self):
        args = type("Args", (), dict(network="", nodes=7, no_update=False))()
        command = join._service_argv(args)
        self.assertIn("--nodes=7", command)
        self.assertNotIn("--network", command)


if __name__ == "__main__":
    unittest.main()
