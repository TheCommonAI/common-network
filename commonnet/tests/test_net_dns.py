import json
import socket
import struct
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from commonnet.net import dns
from commonnet.net.client import BlockClient

NAME = "fresh-tunnel.trycloudflare.com"
real_getaddrinfo = socket.getaddrinfo


def _failing_for_name(host, *args, **kwargs):
    if host == NAME:
        raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided")
    return real_getaddrinfo(host, *args, **kwargs)


class FakeDns:
    """UDP DNS server on localhost answering every A query with 127.0.0.1."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(5)
        self.port = self.sock.getsockname()[1]
        self.queries = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        try:
            while True:
                data, addr = self.sock.recvfrom(512)
                question = data[12:]
                labels, i = [], 0
                while question[i]:
                    labels.append(question[i + 1:i + 1 + question[i]].decode())
                    i += 1 + question[i]
                self.queries.append(".".join(labels))
                header = data[:2] + struct.pack(">HHHHH", 0x8180, 1, 1, 0, 0)
                answer = b"\xc0\x0c" + struct.pack(">HHIH", 1, 1, 30, 4) + socket.inet_aton("127.0.0.1")
                self.sock.sendto(header + question[:i + 5] + answer, addr)
        except OSError:
            pass

    def close(self):
        self.sock.close()


class Echo(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"host": self.headers["Host"], "auth": self.headers.get("Authorization")}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class DnsFallbackTest(unittest.TestCase):
    def setUp(self):
        dns.clear_cache()
        self.dns = FakeDns()
        self.addCleanup(self.dns.close)
        self.http = HTTPServer(("127.0.0.1", 0), Echo)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.port = self.http.server_address[1]
        patcher = mock.patch.object(dns, "FALLBACK_SERVERS", (("127.0.0.1", self.dns.port),))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fallback_when_system_resolver_fails(self):
        client = BlockClient(token="t" * 32, retries=0)
        with mock.patch("socket.getaddrinfo", _failing_for_name):
            reply = client._call("GET", f"http://{NAME}:{self.port}", "/x", None, 5)
        self.assertEqual(reply["host"], f"{NAME}:{self.port}")
        self.assertEqual(reply["auth"], "Bearer " + "t" * 32)
        self.assertEqual(self.dns.queries, [NAME])

    def test_result_is_cached(self):
        with mock.patch("socket.getaddrinfo", _failing_for_name):
            self.assertEqual(dns.resolve(NAME), "127.0.0.1")
            self.assertEqual(dns.resolve(NAME), "127.0.0.1")
        self.assertEqual(self.dns.queries, [NAME])

    def test_both_fail_raises_gaierror(self):
        with mock.patch.object(dns, "FALLBACK_SERVERS", ()), mock.patch("socket.getaddrinfo", _failing_for_name):
            with self.assertRaises(socket.gaierror):
                dns.resolve(NAME)

    def test_system_success_never_uses_fallback(self):
        client = BlockClient(token="t" * 32, retries=0)
        reply = client._call("GET", f"http://localhost:{self.port}", "/x", None, 5)
        self.assertEqual(reply["host"], f"localhost:{self.port}")
        self.assertEqual(dns.resolve("localhost"), "localhost")
        self.assertEqual(self.dns.queries, [])


if __name__ == "__main__":
    unittest.main()
