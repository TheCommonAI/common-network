"""Launcher tests. No cloudflared and no network: the "tunnel" is the local server's own address.

The signal tests run the real main() in a child process, with join.start_tunnel
replaced by a stub. The child gets SIGTERM and SIGINT and must stop the server,
clear the URL in the file and exit 0.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from commonnet.net import tunnel
from commonnet.net.tunnel import LauncherError

REPO_ROOT = tunnel.REPO_ROOT
SLEEPER = [sys.executable, "-c", "import time; time.sleep(600)"]


def wait_until(predicate, timeout=20.0, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def read_record(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def published_url(path: Path):
    return (read_record(path) or {}).get("tunnel_url")


def refused(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return False
    except OSError:
        return True


def quiet(message):
    pass


class RunWithFakeTunnel(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "tunnel.json"

    def test_publishes_the_url_then_clears_it_and_stops_the_server_on_stop(self):
        port = tunnel.free_loopback_port()
        stop = threading.Event()

        def opener(local_url, port_):
            return subprocess.Popen(SLEEPER), local_url

        worker = threading.Thread(target=tunnel.run, args=(["--stub", "--layers", "0", "4"], self.out),
                                  kwargs=dict(open_tunnel=opener, stop=stop, log=quiet, port=port,
                                              health_interval=60))
        worker.start()
        self.assertTrue(wait_until(lambda: published_url(self.out)))
        record = read_record(self.out)
        self.assertEqual(record["tunnel_url"], f"http://127.0.0.1:{port}")
        self.assertEqual(record["layers"], [0, 4])
        self.assertEqual(record["device"], "stub")

        stop.set()
        worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(read_record(self.out)["tunnel_url"])
        self.assertTrue(refused(port), "the block server must be gone after stop")

    def test_an_unreachable_tunnel_publishes_nothing_and_stops_the_server(self):
        port = tunnel.free_loopback_port()
        dead = tunnel.free_loopback_port()  # nothing listens here

        def opener(local_url, port_):
            return subprocess.Popen(SLEEPER), f"http://127.0.0.1:{dead}"

        with patch.object(tunnel._join(), "TUNNEL_WARMUP_TIMEOUT_SECONDS", 0.5):
            with self.assertRaises(LauncherError) as caught:
                tunnel.run(["--stub", "--layers", "0", "2"], self.out, open_tunnel=opener,
                           log=quiet, port=port)
        self.assertIn("warm-up", str(caught.exception))
        self.assertIsNone(published_url(self.out), "a dead URL must never be published")
        self.assertTrue(refused(port))

    def test_a_tunnel_that_stays_down_is_replaced_and_the_new_url_published(self):
        port = tunnel.free_loopback_port()
        local = f"http://127.0.0.1:{port}"
        replacement = f"http://127.0.0.1:{port}/replacement"
        urls = []
        local_is_down = threading.Event()
        real_healthy = tunnel._join().tunnel_is_healthy

        def fake_healthy(url, token, path="/v1/models"):
            if url == local and local_is_down.is_set():
                return False
            return real_healthy(url, token, path)

        def opener(local_url, port_):
            url = local if not urls else replacement
            urls.append(url)
            return subprocess.Popen(SLEEPER), url

        stop = threading.Event()
        with patch.object(tunnel._join(), "tunnel_is_healthy", side_effect=fake_healthy):
            worker = threading.Thread(target=tunnel.run, args=(["--stub", "--layers", "0", "2"], self.out),
                                      kwargs=dict(open_tunnel=opener, stop=stop, log=quiet, port=port,
                                                  health_interval=0.05, grace=0))
            worker.start()
            self.assertTrue(wait_until(lambda: published_url(self.out) == local))
            local_is_down.set()
            self.assertTrue(wait_until(lambda: published_url(self.out) == replacement))
            stop.set()
            worker.join(timeout=20)
        self.assertEqual(urls, [local, replacement])
        self.assertIsNone(published_url(self.out))

    def test_the_block_server_dying_stops_the_launcher_with_an_error(self):
        port = tunnel.free_loopback_port()
        started = []
        real_start = tunnel._start_server

        def recording_start(*args, **kwargs):
            proc = real_start(*args, **kwargs)
            started.append(proc)
            return proc

        def opener(local_url, port_):
            return subprocess.Popen(SLEEPER), local_url

        outcomes = []

        def run():
            # Every exception is recorded, so a failure says what the launcher raised.
            try:
                tunnel.run(["--stub", "--layers", "0", "2"], self.out, open_tunnel=opener, log=quiet,
                           port=port, health_interval=0.05)
                outcomes.append(None)
            except BaseException as e:
                outcomes.append(e)

        with patch.object(tunnel, "_start_server", side_effect=recording_start):
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(wait_until(lambda: published_url(self.out)))
            started[0].kill()
            # Reap the child before the launcher looks at it, so poll() reports the
            # exit rather than racing the kernel's delivery of the signal.
            started[0].wait(timeout=10)
            worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(outcomes), 1, "the launcher thread ended without an outcome")
        self.assertIsInstance(outcomes[0], LauncherError, repr(outcomes[0]))
        self.assertIn("exited", str(outcomes[0]))
        self.assertIsNone(published_url(self.out))
        self.assertTrue(refused(port))


class DefaultOpener(unittest.TestCase):
    def test_calls_join_start_tunnel_with_the_loopback_host_header(self):
        fake_proc = subprocess.Popen(SLEEPER)
        self.addCleanup(fake_proc.kill)
        with patch.object(tunnel._join(), "start_tunnel",
                          return_value=(fake_proc, "https://x.trycloudflare.com")) as start:
            proc, url = tunnel._default_open_tunnel("http://127.0.0.1:50123", 50123)
        start.assert_called_once_with("http://127.0.0.1:50123", "127.0.0.1:50123")
        self.assertIs(proc, fake_proc)
        self.assertEqual(url, "https://x.trycloudflare.com")


WRAPPER = """
import subprocess, sys
sys.path.insert(0, {repo!r})
from commonnet.net import tunnel
join = tunnel._join()
def fake_start_tunnel(target_url, host_header):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    return proc, target_url
join.start_tunnel = fake_start_tunnel
sys.exit(tunnel.main({argv!r}))
"""


class SignalsStopCleanly(unittest.TestCase):
    def run_and_signal(self, signum):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "tunnel.json"
        argv = ["--stub", "--layers", "0", "2", "--out", str(out)]
        code = WRAPPER.format(repo=str(REPO_ROOT), argv=argv)
        child = subprocess.Popen([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.assertTrue(wait_until(lambda: published_url(out), timeout=30),
                        "the launcher never published a URL")
        port = int(published_url(out).rsplit(":", 1)[1])
        self.assertFalse(refused(port), "the block server should be up while published")
        child.send_signal(signum)
        returncode = child.wait(timeout=20)
        stderr = child.stderr.read()
        child.stdout.close()
        child.stderr.close()
        return returncode, read_record(out), port, stderr

    def test_sigterm_clears_the_url_and_stops_the_server(self):
        returncode, record, port, stderr = self.run_and_signal(signal.SIGTERM)
        self.assertEqual(returncode, 0, stderr)
        self.assertIsNone(record["tunnel_url"])
        self.assertTrue(refused(port))

    def test_ctrl_c_clears_the_url_and_stops_the_server(self):
        returncode, record, port, stderr = self.run_and_signal(signal.SIGINT)
        self.assertEqual(returncode, 0, stderr)
        self.assertIsNone(record["tunnel_url"])
        self.assertTrue(refused(port))


class Arguments(unittest.TestCase):
    def test_a_bad_layer_range_is_refused_before_anything_starts(self):
        self.assertEqual(tunnel.main(["--stub", "--layers", "4", "4", "--out", os.devnull]), 2)

    def test_stub_and_block_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            tunnel.main(["--stub", "--block", "x:y", "--layers", "0", "2"])


if __name__ == "__main__":
    unittest.main()
