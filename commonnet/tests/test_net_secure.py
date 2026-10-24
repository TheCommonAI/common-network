"""TLS pinning helpers and the loopback end-to-end: BlockServer over TLS, BlockClient with a pin."""
import http.client
import socket
import tempfile
import threading
import time
import unittest

from commonnet.mesh.records import Identity
from commonnet.net import protocol, secure
from commonnet.net.client import BlockClient, RemoteError, Unreachable
from commonnet.net.protocol import Tensor
from commonnet.net.server import BlockServer, StubBlock

TOKEN = "t" * 32


def tensor(n_bytes: int) -> Tensor:
    return Tensor((1, 1, n_bytes // 2), "float16", bytes(i % 251 for i in range(n_bytes)))


class Fingerprints(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.identity = Identity.generate()

    def test_cert_is_stable_across_calls_and_key_is_private(self):
        import os, stat
        cert, key, fp = secure.make_peer_cert(self.identity, self.dir.name)
        again = secure.make_peer_cert(self.identity, self.dir.name)
        self.assertEqual((cert, key, fp), again)
        self.assertEqual(len(fp), 64)
        self.assertEqual(stat.S_IMODE(os.stat(key).st_mode), 0o600)

    def test_corrupt_cert_is_replaced(self):
        cert, _, fp = secure.make_peer_cert(self.identity, self.dir.name)
        open(cert, "w").write("garbage")
        self.assertNotEqual(secure.make_peer_cert(self.identity, self.dir.name)[2], fp)

    def test_binding_verifies_and_rejects(self):
        _, _, fp = secure.make_peer_cert(self.identity, self.dir.name)
        sig = secure.bind_fingerprint(self.identity, fp)
        self.assertTrue(secure.verify_binding(self.identity.pub_b64, fp, sig))
        other = Identity.generate()
        self.assertFalse(secure.verify_binding(other.pub_b64, fp, sig))
        self.assertFalse(secure.verify_binding(self.identity.pub_b64, "0" * 64, sig))
        self.assertFalse(secure.verify_binding(self.identity.pub_b64, fp, "!!notbase64"))
        self.assertFalse(secure.verify_binding("", fp, sig))

    def test_binding_is_not_a_record_signature(self):
        # domain separation: the same key signing the fingerprint text alone must not verify
        _, _, fp = secure.make_peer_cert(self.identity, self.dir.name)
        raw = self.identity._key.sign(fp.encode("ascii"))
        import base64
        self.assertFalse(secure.verify_binding(self.identity.pub_b64, fp, base64.b64encode(raw).decode()))

    def test_normalize(self):
        fp = "AB" * 32
        self.assertEqual(secure.normalize_fingerprint(":".join(fp[i:i + 2] for i in range(0, 64, 2))), "ab" * 32)
        for bad in ("", "zz" * 32, "ab" * 31, None):
            with self.assertRaises(ValueError):
                secure.normalize_fingerprint(bad)


class LoopbackTls(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        cert, key, self.fp = secure.make_peer_cert(Identity.generate(), self.dir.name)
        self.block = StubBlock(0, 4)
        self.server = BlockServer(self.block, "127.0.0.1", 0, token=TOKEN,
                                  ssl_context=secure.server_ssl_context(cert, key)).start()
        self.addCleanup(self.server.stop)
        self.url = self.server.url

    def client(self, **kw):
        c = BlockClient(token=kw.pop("token", TOKEN), retries=kw.pop("retries", 0), **kw)
        self.addCleanup(c.close_connections)
        return c

    def test_url_is_https(self):
        self.assertTrue(self.url.startswith("https://127.0.0.1:"))

    def test_right_pin_works_and_connection_is_kept_alive(self):
        c = self.client(pin=self.fp)
        self.assertEqual(c.health(self.url)[1], "stub")
        c.open(self.url, "s1")
        c.forward(self.url, "s1", tensor(16), 0)
        c.forward(self.url, "s1", tensor(16), 1)
        reused = [t["reused"] for t in c.timings]
        self.assertEqual(reused, [False, True, True, True])

    def test_pin_as_dict_per_endpoint(self):
        key = f"127.0.0.1:{self.server.port}"
        self.assertEqual(self.client(pin={key: self.fp}).health(self.url)[1], "stub")
        with self.assertRaises(Unreachable):
            self.client(pin={key: "0" * 64}).health(self.url)

    def test_wrong_pin_refused_without_retrying(self):
        c = self.client(pin="0" * 64, retries=3, backoff=5.0)
        started = time.monotonic()
        with self.assertRaises(Unreachable) as cm:
            c.health(self.url)
        self.assertIn("pinned", str(cm.exception))
        self.assertLess(time.monotonic() - started, 3.0)  # no backoff sleeps
        self.assertEqual(self.block.calls, [])

    def test_no_pin_means_system_trust_which_refuses_self_signed(self):
        with self.assertRaises(Unreachable):
            self.client().health(self.url)

    def test_plain_http_to_tls_port_fails_cleanly(self):
        with self.assertRaises(Unreachable):
            self.client().health(self.url.replace("https://", "http://"))
        # and the server is still fine afterwards
        self.assertEqual(self.client(pin=self.fp).health(self.url)[1], "stub")

    def test_plain_bytes_on_tls_port_do_not_wedge_server(self):
        s = socket.create_connection(("127.0.0.1", self.server.port), 2)
        s.sendall(b"GET /health HTTP/1.1\r\n\r\n")
        s.settimeout(2)
        try:
            s.recv(100)
        except OSError:
            pass
        s.close()
        self.assertEqual(self.client(pin=self.fp).health(self.url)[1], "stub")

    def test_stalled_handshake_does_not_block_others(self):
        stalled = socket.create_connection(("127.0.0.1", self.server.port), 2)
        try:
            self.assertEqual(self.client(pin=self.fp).health(self.url)[1], "stub")
        finally:
            stalled.close()

    def test_token_still_required(self):
        with self.assertRaises(RemoteError) as cm:
            self.client(pin=self.fp, token=None).health(self.url)
        self.assertEqual(cm.exception.status, 401)
        with self.assertRaises(RemoteError):
            self.client(pin=self.fp, token="x" * 32).health(self.url)

    def test_700kb_forward_body_and_throughput(self):
        c = self.client(pin=self.fp)
        c.open(self.url, "big")
        body = tensor(700 * 1024)
        started = time.monotonic()
        out = c.forward(self.url, "big", body, 0)
        self.assertEqual(out.data, body.data)
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertGreater(c.timings[-1]["bytes_out"], 700 * 1024)

    def test_concurrent_clients(self):
        errors = []

        def work(i):
            try:
                c = self.client(pin=self.fp)
                c.open(self.url, f"c{i}")
                c.forward(self.url, f"c{i}", tensor(64), 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        [t.start() for t in threads]
        [t.join(10) for t in threads]
        self.assertEqual(errors, [])

    def test_non_loopback_needs_tls_and_token(self):
        with self.assertRaises(ValueError):
            BlockServer(StubBlock(0, 4), "0.0.0.0", 0)
        with self.assertRaises(ValueError):
            BlockServer(StubBlock(0, 4), "0.0.0.0", 0, ssl_context=secure.server_ssl_context(
                *secure.make_peer_cert(Identity.generate(), self.dir.name + "/x")[:2]))

    def test_defaults_unchanged(self):
        plain = BlockServer(StubBlock(0, 4), "127.0.0.1", 0, token=TOKEN).start()
        self.addCleanup(plain.stop)
        self.assertTrue(plain.url.startswith("http://"))
        self.assertEqual(self.client().health(plain.url)[1], "stub")


if __name__ == "__main__":
    unittest.main()
