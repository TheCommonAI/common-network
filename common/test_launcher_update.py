"""Launcher updates must notice same-version UI fixes and preserve working installs."""
import contextlib
import http.server
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

ROOT = Path(__file__).resolve().parents[1]


def archive(bundle):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, body in bundle.items():
            info = tarfile.TarInfo("common-main/" + common.LAUNCHER_FILES[name])
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return stream.getvalue()


class LauncherUpdateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name).resolve()
        self.bundle = {name: (ROOT / source).read_bytes() for name, source in common.LAUNCHER_FILES.items()}
        for name, data in self.bundle.items():
            (self.home / name).write_bytes(data)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(common, "__file__", str(self.home / "common.py")))
        self.stack.enter_context(mock.patch.object(common, "_in_git_work_tree", return_value=False))
        self.stack.enter_context(mock.patch.object(common, "_LAUNCHER_CURRENT", False))
        self.stack.enter_context(mock.patch.object(common, "INSTALL_DIR", self.home))

    def test_unchanged_cli_still_updates_launcher_and_then_reports_current(self):
        (self.home / "join.py").write_bytes(b"# stale banner\n")
        output = io.StringIO()
        with mock.patch.object(common, "_fetch_update_bundle", return_value=self.bundle), contextlib.redirect_stdout(output):
            common.cmd_update()
            common.cmd_update()
        self.assertEqual((self.home / "join.py").read_bytes(), self.bundle["join.py"])
        self.assertEqual((self.home / "join.py.bak").read_bytes(), b"# stale banner\n")
        self.assertIn("Updated launcher", output.getvalue())
        self.assertIn("already up to date", output.getvalue())
        self.assertIn("build ", output.getvalue())

    def test_auto_update_refreshes_siblings_even_when_cli_is_identical(self):
        (self.home / "layers.py").write_bytes(b"# stale progress UI\n")
        with mock.patch.object(common, "_fetch_update_bundle", return_value=self.bundle), mock.patch.object(common, "_exec") as execute:
            common.self_update()
        execute.assert_not_called()
        self.assertTrue(common._LAUNCHER_CURRENT)
        self.assertEqual((self.home / "layers.py").read_bytes(), self.bundle["layers.py"])

    def test_updated_cli_restarts_after_every_sibling_is_installed(self):
        (self.home / "common.py").write_bytes(b"# stale CLI\n")
        (self.home / "join.py").unlink()
        def execute(argv):
            for name, data in self.bundle.items():
                self.assertEqual((self.home / name).read_bytes(), data)
        with mock.patch.object(common, "_fetch_update_bundle", return_value=self.bundle), mock.patch.object(common, "_exec", side_effect=execute) as execute:
            common.self_update()
        execute.assert_called_once()

    def test_partial_archive_and_invalid_python_do_not_change_installed_files(self):
        cases = [archive({"common.py": self.bundle["common.py"]}),
                 archive({**self.bundle, "join.py": b"def invalid syntax"}), b"not an archive"]
        for body in cases:
            with self.subTest(body_size=len(body)), mock.patch.object(common.urllib.request, "urlopen", return_value=io.BytesIO(body)):
                with self.assertRaises(ValueError):
                    common._fetch_update_bundle()
                for name, data in self.bundle.items():
                    self.assertEqual((self.home / name).read_bytes(), data)

    def test_cache_is_bypassed_for_every_fetch(self):
        requests = []
        def fetch(request, **kwargs):
            requests.append(request)
            return io.BytesIO(archive(self.bundle))
        with mock.patch.object(common.urllib.request, "urlopen", side_effect=fetch):
            common._fetch_update_bundle()
            common._fetch_update_bundle()
        self.assertNotEqual(requests[0].full_url, requests[1].full_url)
        self.assertEqual(requests[0].get_header("Cache-control"), "no-cache")

    def test_interrupted_download_reports_failure_and_preserves_install(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = common.http.client.IncompleteRead(b"partial archive", 1024)
        with mock.patch.object(common.urllib.request, "urlopen", return_value=response), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                common.cmd_update()
        self.assertEqual(stopped.exception.code, 1)
        for name, data in self.bundle.items():
            self.assertEqual((self.home / name).read_bytes(), data)

    def test_failed_replacement_rolls_back_all_installed_files(self):
        updated = {name: data + b"\n# updated\n" for name, data in self.bundle.items()}
        original_replace = os.replace
        failed = False
        def replace(source, target):
            nonlocal failed
            if not failed and Path(source).name == "layers.py" and Path(target) == self.home / "layers.py":
                failed = True
                raise OSError("disk full")
            return original_replace(source, target)
        with mock.patch.object(common.os, "replace", side_effect=replace):
            with self.assertRaises(OSError):
                common._install_update_bundle(updated)
        for name, data in self.bundle.items():
            self.assertEqual((self.home / name).read_bytes(), data)

    def test_join_does_not_fetch_stale_raw_files_after_a_successful_update(self):
        args = type("Args", (), dict(no_update=False, auto=False, model=None, region=None, rest=[]))()
        with mock.patch.object(common, "_LAUNCHER_CURRENT", True), \
                mock.patch.object(common.urllib.request, "urlopen", side_effect=AssertionError("unexpected download")), \
                mock.patch.object(common, "_exec") as execute:
            common.cmd_join_or_serve("join", "https://example", args)
        self.assertIn("--no-update", execute.call_args.args[0])

    def test_real_update_command_restores_original_shaded_banner(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(handler):
                handler.send_response(200)
                handler.end_headers()
                handler.wfile.write(archive(self.bundle))
            def log_message(*args):
                pass
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/archive"
                source = self.bundle["common.py"].decode().replace(
                    'UPDATE_BUNDLE_URL = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main"',
                    f"UPDATE_BUNDLE_URL = {url!r}")
                self.bundle["common.py"] = source.encode()
                (self.home / "common.py").write_text(source)
                (self.home / "join.py").write_text("# stale join with missing banner\n")
                result = subprocess.run([sys.executable, str(self.home / "common.py"), "update"], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Updated launcher", result.stdout)
                script = ("import sys; sys.path.insert(0, sys.argv[1]); import join; "
                          "join.run_network_member=lambda *a, **k: print('Joining progress'); "
                          "sys.argv=['join.py', '--no-update', '--no-api']; join.main()")
                result = subprocess.run([sys.executable, "-c", script, str(self.home)], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("░▒▓██████▓▒░", result.stdout)
                self.assertLess(result.stdout.index("░▒▓"), result.stdout.index("Joining progress"))
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
