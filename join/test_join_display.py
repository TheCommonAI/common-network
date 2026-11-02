"""Foreground join UI: real child output, debug toggling, and terminal restoration."""
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers


class JoinDisplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = io.StringIO()

    def display(self, **kwargs):
        return layers.JoinDisplay(output=self.output, input=io.StringIO(),
                                  log_dir=self.directory.name, **kwargs)

    def test_overall_percentage_covers_download_and_connection_without_details(self):
        with self.display() as display:
            display.api = True
            for line in ["1/5 checking this machine: Apple silicon", "peer secret-id: mlx",
                         "2/5 finding the network", "3/5 choosing layers 0-27", "4/5 downloading 4 GB",
                         "shard 1/2: first.safetensors", "first: 50 MB of 100 MB",
                         "first: 100 MB written", "shard 2/2: second.safetensors",
                         "second: 50 MB of 100 MB", "second: 100 MB written", "loading model...",
                         "opening a Cloudflare tunnel...", "up at https://peer.example",
                         "chat API on http://127.0.0.1:11440/v1", "client: 50 MB of 100 MB",
                         "chat API ready"]:
                display.feed(line)
            log = display.path.read_text()
        text = self.output.getvalue()
        self.assertNotIn("secret-id", text)
        self.assertNotIn("safetensors", text)
        self.assertNotIn("https://peer.example", text)
        percentages = [int(line.split(" · ")[1].split("%")[0]) for line in text.splitlines()]
        self.assertEqual(percentages, sorted(set(percentages)))
        self.assertIn(36, percentages)
        self.assertIn(68, percentages)
        self.assertIn(95, percentages)
        self.assertTrue(text.rstrip().endswith("Connected · 100%   Debug: --debug"))
        self.assertIn("secret-id", log)

    def test_debug_reveals_history_then_streams_and_can_be_hidden(self):
        with self.display() as display:
            display.feed("peer diagnostic")
            self.assertNotIn("peer diagnostic", self.output.getvalue())
            display.toggle()
            display.feed("DHT diagnostic")
            self.assertIn("peer diagnostic", self.output.getvalue())
            self.assertIn("DHT diagnostic", self.output.getvalue())
            display.toggle()
            display.feed("hidden again")
            self.assertNotIn("hidden again", self.output.getvalue())

    def test_warnings_and_port_conflict_are_visible_without_debug(self):
        with self.display() as display:
            display.api = True
            display.feed("up at https://peer")
            display.feed("warning: low disk space")
            display.feed("The chat API could not start on port 11440 (Address already in use)")
            self.assertEqual(display.percent, 100)
        self.assertIn("low disk space", self.output.getvalue())
        self.assertIn("--api 11441", self.output.getvalue())

    def test_real_failed_child_exposes_error_tail_and_writes_log(self):
        with self.display() as display:
            result = display.run_command([sys.executable, "-c",
                                         "import sys; print('unexpected backend failure', file=sys.stderr); sys.exit(7)"])
            self.assertEqual(result.returncode, 7)
            self.assertIn("unexpected backend failure", display.path.read_text())
        self.assertIn("unexpected backend failure", self.output.getvalue())

    def test_node_process_output_is_filtered_and_exit_code_preserved(self):
        script = "print('peer hidden'); print('up at https://example'); raise SystemExit(6)"
        with self.display() as display, \
                mock.patch.object(layers, "ensure_source", return_value=Path(self.directory.name)), \
                mock.patch.object(layers, "ensure_runtime", return_value=sys.executable), \
                mock.patch.object(layers, "build_command", return_value=[sys.executable, "-c", script]):
            result = layers.run(None, checks=lambda env: [], nvidia=False, display=display)
        self.assertEqual(result, 6)
        self.assertIn("Connected · 100%", self.output.getvalue())
        self.assertIn("Join stopped", self.output.getvalue())

    @unittest.skipIf(os.name == "nt", "POSIX terminal test")
    def test_d_key_works_without_enter_and_terminal_is_restored(self):
        import pty
        import select
        import termios
        import time
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        with os.fdopen(os.dup(slave), "r") as terminal:
            display = layers.JoinDisplay(output=terminal, input=terminal, log_dir=self.directory.name)
            # The same PTY needs a writable output stream.
            with os.fdopen(os.dup(slave), "w") as output:
                display.output = output
                original = termios.tcgetattr(slave)
                with display:
                    display.feed("hidden diagnostic")
                    os.write(master, b"d")
                    deadline = time.monotonic() + 3
                    data = b""
                    while time.monotonic() < deadline and b"hidden diagnostic" not in data:
                        if select.select([master], [], [], 0.1)[0]:
                            data += os.read(master, 4096)
                    self.assertIn(b"hidden diagnostic", data)
                restored = termios.tcgetattr(slave)
                # macOS sets PENDIN when applying saved attributes.
                restored[3] &= ~getattr(termios, "PENDIN", 0)
                original[3] &= ~getattr(termios, "PENDIN", 0)
                self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
