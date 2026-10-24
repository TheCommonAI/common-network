"""The gateway directory: announce, list, expiry, caps, rate limits, and the /check reachability probe."""
import base64
import hashlib
import json
import os
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from socket import socket
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gateway"))
os.environ["COMMON_GATEWAY_TEST"] = "1"

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402

from app import directory, main  # noqa: E402

b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731


class Peer:
    def __init__(self):
        self.key = Ed25519PrivateKey.generate()
        self.pub = self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.peer = hashlib.sha256(self.pub).hexdigest()[:32]

    def record(self, network="net", ts=None, seq=1, url="https://x.trycloudflare.com", state="up", **extra):
        import time
        rec = {"v": 1, "network": network, "peer": self.peer, "pub": b64(self.pub), "url": url, "token": "",
               "window": [0, 4], "backend": "stub", "state": state, "gib": 1.0, "seq": seq,
               "ts": time.time() if ts is None else ts, **extra}
        body = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()
        rec["sig"] = b64(self.key.sign(body))
        return rec


def reset():
    directory._records.clear()
    directory._hits.clear()


class Announce(unittest.TestCase):
    def setUp(self):
        reset()
        self.c = TestClient(main.app)

    def post(self, rec, ip="1.1.1.1"):
        return self.c.post("/announce", content=json.dumps(rec), headers={"x-forwarded-for": ip})

    def test_announce_then_list_per_network(self):
        a, b = Peer(), Peer()
        self.assertEqual(self.post(a.record("net1")).json(), {"ok": True})
        self.post(b.record("net2"))
        got = self.c.get("/directory", params={"network": "net1"}).json()
        self.assertEqual([r["peer"] for r in got], [a.peer])
        self.assertEqual(self.c.get("/directory", params={"network": "nope"}).json(), [])

    def test_bad_signature_and_tampering_are_rejected(self):
        p = Peer()
        rec = p.record()
        rec["url"] = "https://evil.example"
        self.assertEqual(self.post(rec).status_code, 400)
        rec = p.record()
        rec["peer"] = "0" * 32
        self.assertEqual(self.post(rec).status_code, 400)
        self.assertEqual(self.post({"junk": 1}).status_code, 400)
        self.assertEqual(self.c.get("/directory", params={"network": "net"}).json(), [])

    def test_unknown_fields_and_oversize_are_rejected(self):
        p = Peer()
        self.assertEqual(self.post(p.record(extra_field=1)).status_code, 400)
        big = p.record()
        big["pad"] = "x" * 5000
        self.assertEqual(self.post(big).status_code, 413)

    def test_stale_or_future_or_leaving_records_are_refused(self):
        import time
        p = Peer()
        self.assertEqual(self.post(p.record(ts=time.time() - 120)).status_code, 400)
        self.assertEqual(self.post(p.record(ts=time.time() + 600)).status_code, 400)
        self.assertEqual(self.post(p.record(state="leaving")).status_code, 400)

    def test_newest_record_wins_and_older_seq_does_not_replace(self):
        p = Peer()
        self.post(p.record(seq=5, url="https://new.example"))
        self.assertTrue(self.post(p.record(seq=2, url="https://old.example")).json().get("stale"))
        got = self.c.get("/directory", params={"network": "net"}).json()
        self.assertEqual(got[0]["url"], "https://new.example")

    def test_ttl_expiry(self):
        import time
        p = Peer()
        self.post(p.record())
        for v in directory._records.values():
            v["at"] = time.time() - directory.TTL - 1
        self.assertEqual(self.c.get("/directory", params={"network": "net"}).json(), [])

    def test_optional_dht_and_fp_fields(self):
        p = Peer()
        self.assertEqual(self.post(p.record(dht=7472)).status_code, 200)
        self.assertEqual(self.post(Peer().record(dht=99999)).status_code, 400)
        fp = "ab" * 32
        q = Peer()
        good = q.record(fp=fp, fpsig="")
        del good["sig"]
        good["fpsig"] = b64(q.key.sign(directory.BIND + fp.encode()))
        good["sig"] = b64(q.key.sign(json.dumps(good, sort_keys=True, separators=(",", ":")).encode()))
        self.assertEqual(self.post(good).status_code, 200)
        bad = Peer().record(fp=fp, fpsig=b64(b"x" * 64))
        self.assertEqual(self.post(bad).status_code, 400)

    def test_caps_per_network_and_per_source(self):
        old = (directory.MAX_PER_NETWORK, directory.MAX_PER_SOURCE, directory.ANNOUNCE_PER_MIN)
        directory.MAX_PER_NETWORK, directory.MAX_PER_SOURCE, directory.ANNOUNCE_PER_MIN = 3, 2, 1000
        try:
            self.assertEqual(self.post(Peer().record(), "9.9.9.9").status_code, 200)
            self.assertEqual(self.post(Peer().record(), "9.9.9.9").status_code, 200)
            self.assertEqual(self.post(Peer().record(), "9.9.9.9").status_code, 429)
            self.assertEqual(self.post(Peer().record(), "8.8.8.8").status_code, 200)
            self.assertEqual(self.post(Peer().record(), "7.7.7.7").status_code, 429)  # network full
            self.assertEqual(self.post(Peer().record("other"), "7.7.7.7").status_code, 200)
        finally:
            directory.MAX_PER_NETWORK, directory.MAX_PER_SOURCE, directory.ANNOUNCE_PER_MIN = old

    def test_global_cap(self):
        old = directory.MAX_TOTAL
        directory.MAX_TOTAL = 1
        try:
            self.post(Peer().record(), "2.2.2.2")
            self.assertEqual(self.post(Peer().record("n2"), "3.3.3.3").status_code, 503)
        finally:
            directory.MAX_TOTAL = old

    def test_rate_limit_per_ip(self):
        p = Peer()
        codes = [self.post(p.record(seq=i), "4.4.4.4").status_code for i in range(directory.ANNOUNCE_PER_MIN + 2)]
        self.assertEqual(codes[-1], 429)
        self.assertEqual(self.post(p.record(seq=99), "5.5.5.5").status_code, 200)

    def test_old_routes_are_still_gone(self):
        self.assertEqual(self.c.get("/nodes").status_code, 410)
        self.assertEqual(self.c.post("/v1/chat/completions").status_code, 410)
        self.assertEqual(self.c.get("/health").json()["mode"], "retired")

    def test_forwarded_header_is_ignored_outside_tests(self):
        with mock.patch.object(directory, "TRUST_TEST_HEADER", False), mock.patch.dict(os.environ, {}, clear=True):
            req = Request({"type": "http", "client": ("10.0.0.1", 1234),
                           "headers": [(b"x-forwarded-for", b"6.6.6.6"), (b"x-real-ip", b"7.7.7.7")]})
            self.assertEqual(directory.caller_ip(req), "10.0.0.1")


class RailwayCaller(unittest.TestCase):
    def setUp(self):
        reset()
        self.c = TestClient(main.app)
        self.env = mock.patch.dict(os.environ, {"RAILWAY_ENVIRONMENT": "production"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_documented_real_ip_wins_over_a_multihop_xff_chain(self):
        for address in ("119.15.77.2", "2606:4700:4700::1111"):
            with self.subTest(address=address), mock.patch.object(directory, "_probe", return_value={"reachable": False}) as probe:
                response = self.c.post("/check", json={"port": 7471, "host": "127.0.0.1"},
                                       headers={"x-real-ip": address,
                                                "x-forwarded-for": "1.1.1.1, 152.233.76.10, 79.127.217.66"})
                self.assertEqual(response.status_code, 200)
                probe.assert_called_once_with(address, 7471, None)

    def test_announcement_source_and_limits_use_real_ip(self):
        record = Peer().record()
        headers = {"x-real-ip": "119.15.77.2", "x-forwarded-for": "1.1.1.1, 79.127.217.66"}
        self.assertEqual(self.c.post("/announce", json=record, headers=headers).status_code, 200)
        self.assertEqual(directory._records[("net", record["peer"])]["src"], "119.15.77.2")
        self.assertIn(("announce", "119.15.77.2"), directory._hits)

    def test_missing_or_invalid_real_ip_never_falls_back_to_xff_or_socket(self):
        invalid = (None, "", "999.1.1.1", "1.2.3", "127.0.0.1", "10.0.0.1", "169.254.169.254",
                   "100.64.0.1", "0.0.0.0", "224.0.0.1", "::1", "fc00::1", "fec0::1", "ff02::1", "fe80::1%eth0", "2001:::1",
                   "::ffff:127.0.0.1", "example.com", "1.1.1.1:7471", "[2606:4700::1111]",
                   "1.1.1.1, 8.8.8.8", "1.1.1.1\r\nX-Other: yes")
        for address in invalid:
            with self.subTest(address=address), mock.patch.object(directory, "_probe") as probe:
                headers = {"x-forwarded-for": "8.8.8.8"}
                if address is not None:
                    headers["x-real-ip"] = address
                self.assertEqual(self.c.post("/check", json={"port": 7471}, headers=headers).status_code, 400)
                probe.assert_not_called()
                self.assertEqual(self.c.post("/announce", json=Peer().record(), headers=headers).status_code, 400)

    def test_duplicate_real_ip_headers_are_refused(self):
        with mock.patch.object(directory, "_probe") as probe:
            response = self.c.post("/check", json={"port": 7471},
                                   headers=[("x-real-ip", "1.1.1.1"), ("x-real-ip", "8.8.8.8")])
            self.assertEqual(response.status_code, 400)
            probe.assert_not_called()

    def test_private_socket_peer_cannot_be_probed_outside_explicit_test_mode(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(directory, "TRUST_TEST_HEADER", False), \
                mock.patch.object(directory, "caller_ip", return_value="127.0.0.1"), \
                mock.patch.object(directory, "_probe") as probe:
            self.assertEqual(self.c.post("/check", json={"port": 7471},
                                         headers={"x-real-ip": "8.8.8.8"}).status_code, 400)
            probe.assert_not_called()


class TlsServer:
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()
        c, k = f"{self.dir.name}/c.pem", f"{self.dir.name}/k.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", k, "-out", c,
                        "-subj", "/CN=t", "-days", "1"], check=True, capture_output=True)
        der = ssl.PEM_cert_to_DER_cert(Path(c).read_text())
        self.fp = hashlib.sha256(der).hexdigest()
        self.ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ctx.load_cert_chain(c, k)
        self.sock = socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.alive = True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.alive:
            try:
                conn, _ = self.sock.accept()
                with self.ctx.wrap_socket(conn, server_side=True):
                    pass
            except (OSError, ssl.SSLError):
                pass

    def close(self):
        self.alive = False
        self.sock.close()
        self.dir.cleanup()


class Check(unittest.TestCase):
    def setUp(self):
        reset()
        self.c = TestClient(main.app)
        self.srv = TlsServer()

    def tearDown(self):
        self.srv.close()

    def check(self, body, ip="127.0.0.1"):
        return self.c.post("/check", content=json.dumps(body), headers={"x-forwarded-for": ip})

    def test_reachable_with_and_without_matching_fingerprint(self):
        self.assertEqual(self.check({"port": self.srv.port}).json(), {"reachable": True, "ip": "127.0.0.1"})
        self.assertTrue(self.check({"port": self.srv.port, "fp": self.srv.fp}).json()["reachable"])
        r = self.check({"port": self.srv.port, "fp": "00" * 32}).json()
        self.assertFalse(r["reachable"])

    def test_closed_port_is_unreachable(self):
        s = socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertFalse(self.check({"port": port}).json()["reachable"])

    def test_only_the_callers_address_is_probed(self):
        # The body cannot name another host; a different caller address means the local server is not reached.
        self.assertEqual(self.check({"port": self.srv.port, "host": "127.0.0.1"}, ip="203.0.113.9").json()["reachable"], False)

    def test_port_range_and_shape(self):
        for body in ({"port": 80}, {"port": 70000}, {"port": "x"}, {}, {"port": 5000, "fp": "zz"}):
            self.assertEqual(self.check(body).status_code, 400, body)

    def test_rate_limit(self):
        codes = [self.check({"port": self.srv.port}).status_code for _ in range(directory.CHECK_PER_MIN + 1)]
        self.assertEqual(codes[-1], 429)


if __name__ == "__main__":
    unittest.main()
