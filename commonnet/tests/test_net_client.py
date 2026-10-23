"""Client tests: the client against the real server over localhost, and its retry path.

Nothing here leaves 127.0.0.1. The endpoints that misbehave are small local stubs.
"""
import json
import socket
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from commonnet.net import protocol
from commonnet.net.client import BlockClient, RemoteError, Unreachable
from commonnet.net.protocol import ProtocolError, Tensor
from commonnet.net.server import BlockServer, StubBlock
from commonnet.tests.test_net_server import hidden_tensor


class Served(unittest.TestCase):
    def setUp(self):
        self.block = StubBlock(0, 8)
        self.server = BlockServer(self.block, "127.0.0.1", 0).start()
        self.endpoint = self.server.url
        self.sleeps = []
        self.client = BlockClient(retries=3, backoff=0.1, sleep=self.sleeps.append)

    def tearDown(self):
        self.server.stop()


class RoundTrip(Served):
    def test_health_open_forward_close(self):
        self.assertEqual(self.client.health(self.endpoint), ((0, 8), "stub"))
        self.client.open(self.endpoint, "job-1")
        hidden = hidden_tensor(seed=9, shape=(1, 2, 32), dtype="float32")
        out = self.client.forward(self.endpoint, "job-1", hidden, 0)
        self.assertEqual(out, hidden)
        self.client.close(self.endpoint, "job-1")
        self.assertEqual(self.sleeps, [])  # nothing failed, so nothing was retried

    def test_forward_against_a_closed_session_raises_remote_error_404(self):
        with self.assertRaises(RemoteError) as caught:
            self.client.forward(self.endpoint, "never-opened", hidden_tensor(), 0)
        self.assertEqual(caught.exception.status, 404)
        self.assertEqual(self.sleeps, [])  # a 404 is an answer, not a transient failure

    def test_trailing_slash_on_the_endpoint_is_accepted(self):
        self.client.open(self.endpoint + "/", "s")
        self.assertEqual(self.block.calls[-1], ("open", "s", None))


class ClientValidation(Served):
    def test_endpoint_must_be_http(self):
        for bad in ("127.0.0.1:8000", "ftp://x", "", None):
            with self.subTest(endpoint=bad):
                with self.assertRaises(ValueError):
                    self.client.health(bad)

    def test_forward_takes_a_tensor(self):
        with self.assertRaises(TypeError):
            self.client.forward(self.endpoint, "s", [[1.0]], 0)

    def test_forward_checks_the_session_id_and_start_pos_before_sending(self):
        with self.assertRaises(ProtocolError):
            self.client.forward(self.endpoint, "../bad", hidden_tensor(), 0)
        with self.assertRaises(ProtocolError):
            self.client.forward(self.endpoint, "s", hidden_tensor(), -1)
        self.assertEqual(self.block.calls, [])

    def test_request_over_the_client_limit_is_refused_before_sending(self):
        small = BlockClient(retries=0, max_request_bytes=1024)
        big = hidden_tensor(shape=(1, 64, 128))
        with self.assertRaises(ValueError):
            small.forward(self.endpoint, "s", big, 0)

    def test_reply_with_a_different_shape_is_rejected(self):
        self.client.open(self.endpoint, "s")
        original = self.block.forward
        self.block.forward = lambda sid, hidden, pos: Tensor((1, 1, 4), hidden.dtype, bytes(8))
        try:
            with self.assertRaises(RemoteError) as caught:
                self.client.forward(self.endpoint, "s", hidden_tensor(), 0)
        finally:
            self.block.forward = original
        self.assertEqual(caught.exception.status, 500)  # the server caught it first


class SilentEndpoint:
    """A TCP listener that accepts connections and never answers. Counts the accepts."""

    def __init__(self):
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.listen(16)
        self.accepted = []
        self.held = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.listener.getsockname()[1]}"

    def _run(self):
        self.listener.settimeout(0.1)
        while not self._stop.is_set():
            try:
                conn, _ = self.listener.accept()
            except (socket.timeout, OSError):
                continue
            self.accepted.append(conn)  # kept open, never answered

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        for conn in self.accepted:
            conn.close()
        self.listener.close()


class RetryPath(unittest.TestCase):
    def test_an_endpoint_that_never_answers_is_retried_then_unreachable(self):
        silent = SilentEndpoint()
        self.addCleanup(silent.close)
        sleeps = []
        client = BlockClient(retries=2, backoff=0.01, sleep=sleeps.append)
        with self.assertRaises(Unreachable) as caught:
            client.forward(silent.endpoint, "s", hidden_tensor(), 0, timeout=0.2)
        self.assertIn("3 times", str(caught.exception))
        self.assertEqual(len(sleeps), 2)  # a sleep between attempts, none after the last
        self.assertEqual(sleeps, [0.01, 0.02])
        self.assertEqual(len(silent.accepted), 3)

    def test_nothing_listening_is_retried_then_unreachable(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()  # nothing listens here now
        sleeps = []
        client = BlockClient(retries=3, backoff=0.01, sleep=sleeps.append)
        with self.assertRaises(Unreachable):
            client.health(f"http://127.0.0.1:{port}", timeout=1)
        self.assertEqual(len(sleeps), 3)

    def test_zero_retries_means_one_attempt(self):
        silent = SilentEndpoint()
        self.addCleanup(silent.close)
        sleeps = []
        client = BlockClient(retries=0, sleep=sleeps.append)
        with self.assertRaises(Unreachable):
            client.open(silent.endpoint, "s", timeout=0.2)
        self.assertEqual(sleeps, [])
        self.assertEqual(len(silent.accepted), 1)


class FlakyFront(BaseHTTPRequestHandler):
    """Forwards every request to the real server. The first /forward is run for real,
    and then its reply is replaced by a 502, as a tunnel does when the far end drops it."""

    real = None
    dropped = False
    calls = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self._pass("GET", None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        FlakyFront.calls.append(self.path)
        if self.path == "/forward" and not FlakyFront.dropped:
            FlakyFront.dropped = True
            _fetch("POST", FlakyFront.real + self.path, body)  # the real server runs the step
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._pass("POST", body)

    def _pass(self, method, body):
        status, data = _fetch(method, FlakyFront.real + self.path, body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _fetch(method, url, body):
    request = urllib.request.Request(url, data=body, method=method,
                                     headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.read()


class RetryIsSafe(unittest.TestCase):
    def test_a_dropped_forward_reply_is_retried_without_running_the_block_twice(self):
        block = StubBlock(0, 4)
        real = BlockServer(block, "127.0.0.1", 0).start()
        self.addCleanup(real.stop)
        FlakyFront.real = real.url
        FlakyFront.dropped = False
        FlakyFront.calls = []
        front = ThreadingHTTPServer(("127.0.0.1", 0), FlakyFront)
        front.daemon_threads = True
        thread = threading.Thread(target=front.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (front.shutdown(), front.server_close()))

        client = BlockClient(retries=2, backoff=0, sleep=lambda s: None)
        endpoint = f"http://127.0.0.1:{front.server_address[1]}"
        client.open(endpoint, "s")
        hidden = hidden_tensor(seed=3)
        out = client.forward(endpoint, "s", hidden, 0)

        self.assertEqual(out, hidden)
        self.assertTrue(FlakyFront.dropped)
        self.assertEqual(FlakyFront.calls.count("/forward"), 2)  # the dropped one, then the retry
        forwards = [call for call in block.calls if call[0] == "forward"]
        self.assertEqual(len(forwards), 1)  # the block ran once; the retry got the saved reply


class StatusRetries(unittest.TestCase):
    def test_503_is_retried_and_then_succeeds(self):
        served = BlockServer(StubBlock(0, 2), "127.0.0.1", 0).start()
        self.addCleanup(served.stop)
        state = {"left": 2}

        class Busy(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if state["left"] > 0:
                    state["left"] -= 1
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                with urllib.request.urlopen(served.url + self.path, timeout=5) as response:
                    data = response.read()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        front = ThreadingHTTPServer(("127.0.0.1", 0), Busy)
        thread = threading.Thread(target=front.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (front.shutdown(), front.server_close()))
        sleeps = []
        client = BlockClient(retries=3, backoff=0.01, sleep=sleeps.append)
        layers, device = client.health(f"http://127.0.0.1:{front.server_address[1]}")
        self.assertEqual((layers, device), ((0, 2), "stub"))
        self.assertEqual(len(sleeps), 2)

    def test_a_block_failure_500_is_not_retried(self):
        block = StubBlock(0, 2)
        served = BlockServer(block, "127.0.0.1", 0).start()
        self.addCleanup(served.stop)
        sleeps = []
        client = BlockClient(retries=3, backoff=0.01, sleep=sleeps.append)
        client.open(served.url, "s")

        def broken(sid, hidden, pos):
            raise RuntimeError("cuda error")

        block.forward = broken
        with self.assertRaises(RemoteError) as caught:
            client.forward(served.url, "s", hidden_tensor(), 0)
        self.assertEqual(caught.exception.status, 500)
        self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main()


class KeepAlive(Served):
    def _forward(self, n=1):
        self.client.open(self.endpoint, "k")
        for i in range(n):
            self.client.forward(self.endpoint, "k", hidden_tensor(seed=i, shape=(1, 1, 32), dtype="float32"), i)

    def test_calls_after_the_first_reuse_the_connection(self):
        self._forward(3)
        reused = [t["reused"] for t in self.client.timings]
        self.assertEqual(reused, [False, True, True, True])  # open, then three forwards

    def test_forward_timings_carry_server_compute_time(self):
        self._forward(1)
        row = [t for t in self.client.timings if t["path"] == "/forward"][0]
        self.assertIsNotNone(row["server_ms"])
        self.assertLessEqual(row["server_ms"], row["total_ms"])
        self.assertGreater(row["bytes_out"], 0)
        self.assertGreater(row["bytes_in"], 0)

    def test_a_connection_the_server_dropped_is_replaced_without_a_retry(self):
        self._forward(1)
        for conn in list(self.client._idle.values()):
            conn.sock.close()  # as if the far side timed it out while idle
        self.client.forward(self.endpoint, "k", hidden_tensor(seed=5, shape=(1, 1, 32), dtype="float32"), 1)
        self.assertEqual(self.sleeps, [])  # no backoff was needed

    def test_a_refused_oversize_request_closes_the_connection_and_the_next_call_works(self):
        small = BlockServer(StubBlock(0, 8), "127.0.0.1", 0, max_request_bytes=2048).start()
        self.addCleanup(small.stop)
        raw = http_post_oversize(small.url)
        self.assertEqual(raw, 413)
        self.assertEqual(self.client.health(small.url), ((0, 8), "stub"))

    def test_close_connections_empties_the_pool(self):
        self._forward(1)
        self.client.close_connections()
        self.assertEqual(self.client._idle, {})


def http_post_oversize(url):
    import http.client
    host, port = url[len("http://"):].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    conn.request("POST", "/forward", body=b"x" * 5000, headers={"Content-Type": "application/json"})
    status = conn.getresponse().status
    conn.close()
    return status
