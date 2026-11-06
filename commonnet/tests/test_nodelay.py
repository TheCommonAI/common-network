"""TCP_NODELAY on every connection that carries tokens.

Nagle holds small request/reply messages back (~80 ms per token on the Kaggle
split-Bonsai link). Client connections set it after connect (and after TLS wrap,
which is the tunnel/direct hop). The block server and the local chat API set the
stdlib's disable_nagle_algorithm for each accepted connection.
"""
import socket
import tempfile
import threading
import unittest
from types import SimpleNamespace

from commonnet.mesh.records import Identity
from commonnet.net import client as client_mod
from commonnet.net import secure, server as server_mod
from commonnet.net.client import BlockClient
from commonnet.net.server import BlockServer, StubBlock
from commonnet.serve import make_handler


class NoDelayTests(unittest.TestCase):
    def _nagle_off(self, sock):
        self.assertNotEqual(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 0)  # 0 means Nagle is on

    def test_a_client_connection_has_nagle_turned_off(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        conn = client_mod._HTTPConnection("127.0.0.1", listener.getsockname()[1], timeout=5)
        self.addCleanup(conn.close)
        conn.connect()
        self._nagle_off(conn.sock)

    def test_nagle_is_turned_off_after_tls_wrap(self):
        """wrap_socket may replace the socket object; nodelay must land on what HTTP writes."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        port = listener.getsockname()[1]

        class Ctx:
            def wrap_socket(self, sock, server_hostname=None):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)  # wrap "resets" it
                return sock

        conn = SimpleNamespace(host="127.0.0.1", port=port, timeout=5, source_address=None,
                               public_only=False, _context=Ctx(), sock=None)
        err = []

        def connect():
            try:
                client_mod._connect_resolved(conn, True)
            except BaseException as e:
                err.append(e)

        worker = threading.Thread(target=connect)
        worker.start()
        peer, _ = listener.accept()
        self.addCleanup(peer.close)
        worker.join(5)
        self.assertFalse(err)
        self.assertIsNotNone(conn.sock)
        self.addCleanup(conn.sock.close)
        self._nagle_off(conn.sock)

    def test_a_pooled_block_client_keeps_nagle_off(self):
        server = BlockServer(StubBlock(0, 4), "127.0.0.1", 0).start()
        self.addCleanup(server.stop)
        client = BlockClient(token=None, retries=0)
        self.addCleanup(client.close_connections)
        client.health(server.url)
        conn = next(iter(client._idle.values()))
        self._nagle_off(conn.sock)

    def test_an_https_block_client_connection_has_nagle_turned_off(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cert, key, fp = secure.make_peer_cert(Identity.generate(), directory.name)
        server = BlockServer(StubBlock(0, 4), "127.0.0.1", 0,
                             ssl_context=secure.server_ssl_context(cert, key)).start()
        self.addCleanup(server.stop)
        client = BlockClient(token=None, retries=0, pin=fp)
        self.addCleanup(client.close_connections)
        client.health(server.url)
        conn = next(iter(client._idle.values()))
        self._nagle_off(conn.sock)

    def test_a_server_connection_has_nagle_turned_off(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        a = socket.create_connection(listener.getsockname())
        self.addCleanup(a.close)
        b, _ = listener.accept()
        self.addCleanup(b.close)
        handler = server_mod._Handler.__new__(server_mod._Handler)
        handler.request = b
        handler.client_address = ("127.0.0.1", 0)
        handler.server = None
        handler.setup()
        self.addCleanup(handler.finish)
        self._nagle_off(handler.connection)

    def test_the_local_chat_api_turns_nagle_off(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        a = socket.create_connection(listener.getsockname())
        self.addCleanup(a.close)
        b, _ = listener.accept()
        self.addCleanup(b.close)
        handler_cls = make_handler(SimpleNamespace(model_name="T"))
        handler = handler_cls.__new__(handler_cls)
        handler.request = b
        handler.client_address = ("127.0.0.1", 0)
        handler.server = None
        handler.setup()
        self.addCleanup(handler.finish)
        self._nagle_off(handler.connection)


if __name__ == "__main__":
    unittest.main()
