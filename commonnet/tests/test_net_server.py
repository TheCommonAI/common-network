"""Server tests: the HTTP interface over the stub block, on 127.0.0.1 only."""
import http.client
import json
import socket
import threading
import unittest
import urllib.error
import urllib.request

from commonnet.net import protocol
from commonnet.net.protocol import Tensor
from commonnet.net.server import BlockServer, StubBlock


def hidden_tensor(seed: int = 0, shape=(1, 1, 8), dtype="float16") -> Tensor:
    count = protocol.numel(shape) * protocol.DTYPE_WIDTH[dtype]
    return Tensor(shape, dtype, bytes((seed + i) % 256 for i in range(count)))


def post(url: str, path: str, body: dict, timeout: float = 5):
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def post_error(url: str, path: str, body: dict):
    try:
        post(url, path, body)
    except urllib.error.HTTPError as e:
        with e:
            return e.code, json.loads(e.read())
    raise AssertionError(f"{path} did not fail")


class ServerCase(unittest.TestCase):
    max_request_bytes = protocol.MAX_REQUEST_BYTES

    def setUp(self):
        self.block = StubBlock(0, 4)
        self.server = BlockServer(self.block, "127.0.0.1", 0,
                                  max_request_bytes=self.max_request_bytes).start()
        self.url = self.server.url

    def tearDown(self):
        self.server.stop()


class Health(ServerCase):
    def test_health_reports_layers_and_device(self):
        with urllib.request.urlopen(self.url + "/health", timeout=5) as response:
            body = json.loads(response.read())
        self.assertEqual(body, {"ok": True, "layers": [0, 4], "device": "stub"})

    def test_binds_loopback_only(self):
        self.assertEqual(self.url, f"http://127.0.0.1:{self.server.port}")
        with self.assertRaises(ValueError):
            BlockServer(StubBlock(0, 1), "0.0.0.0", 0)
        with self.assertRaises(ValueError):
            BlockServer(StubBlock(0, 1), "192.168.1.5", 0)


class SessionLifecycle(ServerCase):
    def test_open_forward_close(self):
        status, body = post(self.url, "/session", {"id": "s1"})
        self.assertEqual((status, body), (200, {"ok": True}))

        hidden = hidden_tensor()
        status, reply = post(self.url, "/forward", protocol.encode_forward_request("s1", hidden, 0))
        self.assertEqual(status, 200)
        self.assertEqual(protocol.decode_forward_reply(reply), hidden)  # the stub is the identity

        status, body = post(self.url, "/close", {"id": "s1"})
        self.assertEqual((status, body), (200, {"ok": True}))
        self.assertEqual(self.block.calls, [("open", "s1", None), ("forward", "s1", 0), ("close", "s1", None)])

    def test_forward_before_open_is_404(self):
        status, body = post_error(self.url, "/forward", protocol.encode_forward_request("nope", hidden_tensor(), 0))
        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])
        self.assertEqual(self.block.calls, [])

    def test_forward_after_close_is_404(self):
        post(self.url, "/session", {"id": "s1"})
        post(self.url, "/close", {"id": "s1"})
        status, _ = post_error(self.url, "/forward", protocol.encode_forward_request("s1", hidden_tensor(), 0))
        self.assertEqual(status, 404)

    def test_open_and_close_are_idempotent(self):
        post(self.url, "/session", {"id": "s1"})
        status, body = post(self.url, "/session", {"id": "s1"})
        self.assertEqual((status, body), (200, {"ok": True}))
        post(self.url, "/close", {"id": "s1"})
        status, body = post(self.url, "/close", {"id": "s1"})
        self.assertEqual((status, body), (200, {"ok": True}))
        opens = [call for call in self.block.calls if call[0] == "open"]
        self.assertEqual(len(opens), 1)

    def test_a_repeated_forward_is_answered_without_running_the_block_again(self):
        post(self.url, "/session", {"id": "s1"})
        hidden = hidden_tensor(seed=5)
        body = protocol.encode_forward_request("s1", hidden, 3)
        _, first = post(self.url, "/forward", body)
        _, second = post(self.url, "/forward", body)
        self.assertEqual(first, second)
        forwards = [call for call in self.block.calls if call[0] == "forward"]
        self.assertEqual(len(forwards), 1)

    def test_a_different_step_at_the_same_position_runs_the_block(self):
        post(self.url, "/session", {"id": "s1"})
        post(self.url, "/forward", protocol.encode_forward_request("s1", hidden_tensor(seed=1), 3))
        post(self.url, "/forward", protocol.encode_forward_request("s1", hidden_tensor(seed=2), 3))
        forwards = [call for call in self.block.calls if call[0] == "forward"]
        self.assertEqual(len(forwards), 2)

    def test_sessions_are_independent(self):
        post(self.url, "/session", {"id": "a"})
        post(self.url, "/session", {"id": "b"})
        post(self.url, "/close", {"id": "a"})
        status, _ = post(self.url, "/forward", protocol.encode_forward_request("b", hidden_tensor(), 0))
        self.assertEqual(status, 200)

    def test_block_failure_is_500_and_session_survives(self):
        post(self.url, "/session", {"id": "s1"})
        original = self.block.forward

        def broken(session_id, hidden, start_pos):
            raise RuntimeError("out of memory")

        self.block.forward = broken
        try:
            status, body = post_error(self.url, "/forward", protocol.encode_forward_request("s1", hidden_tensor(), 0))
        finally:
            self.block.forward = original
        self.assertEqual(status, 500)
        self.assertIn("out of memory", body["error"])
        status, _ = post(self.url, "/forward", protocol.encode_forward_request("s1", hidden_tensor(), 0))
        self.assertEqual(status, 200)

    def test_block_returning_the_wrong_shape_is_500(self):
        post(self.url, "/session", {"id": "s1"})
        self.block.forward = lambda sid, hidden, pos: Tensor((1, 1, 4), hidden.dtype, bytes(8))
        status, body = post_error(self.url, "/forward", protocol.encode_forward_request("s1", hidden_tensor(), 0))
        self.assertEqual(status, 500)
        self.assertIn("shape", body["error"])


class Validation(ServerCase):
    def raw(self, method, path, body=b"", headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def test_bad_json_is_400(self):
        status, _ = self.raw("POST", "/session", b"{not json", {"Content-Length": "9"})
        self.assertEqual(status, 400)

    def test_malformed_envelopes_are_400(self):
        good = protocol.encode_forward_request("s1", hidden_tensor(), 0)
        post(self.url, "/session", {"id": "s1"})
        for bad in (dict(good, start_pos=-1), dict(good, shape=[1, 8]), dict(good, dtype="float64"),
                    dict(good, extra=True), {"id": "s1"}):
            with self.subTest(body=sorted(bad)):
                status, _ = post_error(self.url, "/forward", bad)
                self.assertEqual(status, 400)

    def test_unknown_path_is_404_and_wrong_method_is_405(self):
        status, _ = self.raw("GET", "/nope")
        self.assertEqual(status, 404)
        status, _ = self.raw("GET", "/forward")
        self.assertEqual(status, 405)
        status, _ = self.raw("POST", "/health", b"{}", {"Content-Length": "2"})
        self.assertEqual(status, 405)

    def test_query_strings_are_ignored_for_routing(self):
        status, _ = self.raw("GET", "/health?x=1")
        self.assertEqual(status, 200)

    def test_chunked_body_is_refused(self):
        status, _ = self.raw("POST", "/session", b"{}", {"Transfer-Encoding": "chunked"})
        self.assertEqual(status, 411)

    def test_missing_content_length_is_refused(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        conn.putrequest("POST", "/session", skip_accept_encoding=True)
        conn.endheaders()
        response = conn.getresponse()
        self.assertEqual(response.status, 411)
        conn.close()

    def test_short_body_is_400(self):
        conn = socket.create_connection(("127.0.0.1", self.server.port), timeout=5)
        conn.sendall(b'POST /session HTTP/1.0\r\nContent-Length: 50\r\n\r\n{"id":"x"}')
        conn.shutdown(socket.SHUT_WR)
        reply = conn.recv(4096)
        conn.close()
        self.assertIn(b" 400 ", reply.split(b"\r\n", 1)[0])


class RequestSizeLimit(ServerCase):
    max_request_bytes = 1024

    def test_oversized_body_is_refused_before_it_is_read(self):
        # Declares 10 MB and sends none of it. A server that read the body would
        # wait here; the refusal must come back at once, from the header alone.
        conn = socket.create_connection(("127.0.0.1", self.server.port), timeout=5)
        conn.sendall(b"POST /forward HTTP/1.0\r\nContent-Length: 10000000\r\n"
                     b"Content-Type: application/json\r\n\r\n")
        reply = conn.recv(4096)
        conn.close()
        self.assertIn(b" 413 ", reply.split(b"\r\n", 1)[0])
        self.assertEqual(self.block.calls, [])

    def test_body_at_the_limit_is_accepted(self):
        post(self.url, "/session", {"id": "s1"})
        small = protocol.encode_forward_request("s1", hidden_tensor(shape=(1, 1, 8)), 0)
        self.assertLess(len(json.dumps(small)), self.max_request_bytes)
        status, _ = post(self.url, "/forward", small)
        self.assertEqual(status, 200)


class Concurrency(ServerCase):
    def test_parallel_sessions_are_all_served_and_the_block_is_never_entered_twice(self):
        self.block.delay = 0.005
        errors = []

        def worker(index):
            try:
                sid = f"w{index}"
                post(self.url, "/session", {"id": sid})
                for step in range(3):
                    status, _ = post(self.url, "/forward",
                                     protocol.encode_forward_request(sid, hidden_tensor(seed=index + step), step))
                    if status != 200:
                        errors.append((sid, status))
                post(self.url, "/close", {"id": sid})
            except Exception as e:  # collected for the main thread
                errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertFalse(self.block.overlapped)
        forwards = [call for call in self.block.calls if call[0] == "forward"]
        self.assertEqual(len(forwards), 24)


class StubBlockValidation(unittest.TestCase):
    def test_stub_rejects_an_empty_layer_range(self):
        with self.assertRaises(protocol.ProtocolError):
            StubBlock(4, 4)


if __name__ == "__main__":
    unittest.main()
