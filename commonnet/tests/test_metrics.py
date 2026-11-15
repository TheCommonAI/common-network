"""Timing definitions, outcomes, expiry, reporting failures and real chat HTTP instrumentation."""
import http.client
import json
import threading
import time
import unittest

from commonnet.metrics import MeasuredStream, Metrics, Reporter
from commonnet.mesh.records import Identity
from commonnet.serve import make_server
from commonnet.tests.test_serve import FakeBackend


class Timings(unittest.TestCase):
    def test_decode_excludes_first_token_and_observations_are_recorded_once(self):
        clock = [0]
        metrics = Metrics(clock=lambda: 1000)
        source = iter([1, 2, 3])
        stream = MeasuredStream(source, metrics, 0, clock=lambda: clock[0])
        for stamp in (0.2, 0.3, 0.5):
            clock[0] = stamp
            next(stream)
        clock[0] = 0.55
        with self.assertRaises(StopIteration):
            next(stream)
        row = metrics.snapshot()[0]
        self.assertEqual(row["output_tokens"], 3)
        self.assertEqual(row["decode_tokens"], 2)
        self.assertEqual(row["decode_ms"], 300)
        self.assertEqual(row["ttft_ms"], 200)
        self.assertEqual(row["total_ms"], 550)
        self.assertEqual(row["completed"], 1)

    def test_cancellation_failure_and_single_token_are_separate(self):
        metrics = Metrics(clock=lambda: 1000)
        class Source:
            def __init__(self): self.closed = False
            def __next__(self): return 1
            def close(self): self.closed = True
        source = Source()
        stream = MeasuredStream(source, metrics, 0, clock=lambda: 1)
        next(stream)
        stream.close()
        stream.close()
        self.assertTrue(source.closed)
        def failed():
            raise RuntimeError("peer gone")
            yield
        stream = MeasuredStream(failed(), metrics, 0, clock=lambda: 1)
        with self.assertRaises(RuntimeError):
            next(stream)
        stream.close()
        stream = MeasuredStream(iter([1]), metrics, 0, clock=lambda: 1)
        self.assertEqual(list(stream), [1])
        row = metrics.snapshot()[0]
        self.assertEqual((row["requests"], row["completed"], row["failed"], row["cancelled"]), (3, 1, 1, 1))
        self.assertEqual((row["output_tokens"], row["decode_tokens"]), (1, 0))

    def test_buckets_expire(self):
        now = [1000]
        metrics = Metrics(clock=lambda: now[0])
        metrics.record(outcome="completed", count=2, first=1, last=2, started=0, ended=2)
        now[0] = 1301
        self.assertEqual(metrics.snapshot(), [])

    def test_reporting_failure_does_not_stop_the_node_and_stop_sends_leaving(self):
        phase, sent, arrived = ["up"], [], threading.Event()
        def post(body):
            sent.append(body)
            arrived.set()
            raise OSError("directory offline")
        snapshot = lambda: dict(network="net", phase=phase[0], backend="mlx", gib=4, layers=None, assigned_layers=None)
        reporter = Reporter(Identity.generate(), "https://example.invalid", snapshot, Metrics(), post=post, interval=100)
        reporter.start()
        self.assertTrue(arrived.wait(2))
        phase[0] = "leaving"
        reporter.stop()
        self.assertFalse(reporter._thread.is_alive())
        self.assertEqual(sent[-1]["phase"], "leaving")
        self.assertLess(sent[0]["seq"], sent[-1]["seq"])
        self.assertTrue(all("sig" in report for report in sent))

    def test_real_plain_and_streaming_http_requests_are_measured(self):
        metrics = Metrics()
        backend = FakeBackend()
        backend.delay = 0.01
        server = make_server(backend, 0, metrics=metrics)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for streaming in (False, True):
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                body = json.dumps({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 3, "stream": streaming})
                conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                conn.close()
            rows = metrics.snapshot()
            self.assertEqual(sum(r["completed"] for r in rows), 2)
            self.assertEqual(sum(r["output_tokens"] for r in rows), 6)
            self.assertEqual(sum(r["decode_tokens"] for r in rows), 4)
            self.assertGreater(sum(r["decode_ms"] for r in rows), 20)
            self.assertNotIn("hello", json.dumps(rows))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_http_failure_and_disconnection_do_not_publish_successful_speed_samples(self):
        from commonnet.net.client import Unreachable
        import socket
        metrics = Metrics()
        backend = FakeBackend()
        server = make_server(backend, 0, metrics=metrics)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend.fail_after, backend.error = 1, Unreachable("peer gone")
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            body = json.dumps({"messages": [{"role": "user", "content": "hello"}], "stream": True})
            conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertIn(b"peer_failed", response.read())
            conn.close()
            backend.fail_after, backend.error, backend.delay, backend.script = None, None, .02, [1] * 80
            with socket.create_connection(("127.0.0.1", server.server_port)) as sock:
                sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                             b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body.encode())
                sock.recv(200)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not sum(r["cancelled"] for r in metrics.snapshot()):
                time.sleep(.01)
            rows = metrics.snapshot()
            self.assertEqual(sum(r["failed"] for r in rows), 1)
            self.assertEqual(sum(r["cancelled"] for r in rows), 1)
            self.assertEqual(sum(r["completed"] for r in rows), 0)
            self.assertEqual(sum(r["decode_tokens"] for r in rows), 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()


class ServedWork(unittest.TestCase):
    """What a node computed for routes: positions through its layers and their time, counted once per step."""

    def test_forward_steps_count_positions_and_retries_count_once(self):
        from commonnet.net.server import BlockServer, StubBlock
        from commonnet.tests.test_net_server import hidden_tensor
        metrics = Metrics()
        server = BlockServer(StubBlock(0, 8), "127.0.0.1", 0, on_forward=metrics.record_work)
        server.open("s1")
        prefill = hidden_tensor(1, shape=(1, 5, 8))
        server.forward("s1", prefill, 0)
        server.forward("s1", prefill, 0)  # a retry of the same step returns the cached reply
        server.forward("s1", hidden_tensor(2), 5)
        buckets = metrics.snapshot()
        self.assertEqual(sum(b["served_tokens"] for b in buckets), 6)
        self.assertGreaterEqual(sum(b["served_compute_ms"] for b in buckets), 0)

    def test_bad_work_values_are_ignored(self):
        metrics = Metrics()
        for tokens, ms in ((0, 1.0), (-3, 1.0), (2.5, 1.0), (2, -1.0), (2, "x")):
            metrics.record_work(tokens, ms)
        self.assertEqual(metrics.snapshot(), [])

    def test_report_names_the_device_and_the_device_vouches_for_the_peer(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from commonnet.metrics import DEVICE_BIND
        import base64
        peer, device = Identity.generate(), Identity.generate()
        snapshot = lambda: {"network": "n", "phase": "up", "backend": "mlx", "gib": 1.0, "layers": None,
                            "assigned_layers": None}
        payload = Reporter(peer, "http://unused", snapshot, Metrics(), device=device).payload()
        self.assertEqual((payload["v"], payload["peer"], payload["device"]), (4, peer.peer, device.peer))
        Ed25519PublicKey.from_public_bytes(device.pub).verify(base64.b64decode(payload["device_sig"]),
                                                              DEVICE_BIND + peer.pub)
        alone = Reporter(peer, "http://unused", snapshot, Metrics()).payload()
        self.assertEqual(alone["device"], peer.peer)
