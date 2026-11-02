"""The Common CLI forwards numeric preset flags, including a capitalised Join verb."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


class PresetForwarding(unittest.TestCase):
    def test_join_capitalisation_and_count_are_forwarded(self):
        for verb in ("join", "Join"):
            with self.subTest(verb=verb), mock.patch.object(sys, "argv", ["common", "--no-update", verb, "--7"]), \
                    mock.patch.object(common, "cmd_join_or_serve") as run:
                common.main()
                self.assertEqual(run.call_args.args[0], "join")
                self.assertEqual(run.call_args.args[2].rest, ["--7"])


if __name__ == "__main__":
    unittest.main()
