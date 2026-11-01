"""TCP_NODELAY on every connection that carries tokens: client connections and the servers' accepted connections.
Without it, Nagle's algorithm holds small messages back, which cost about 80 ms per token on a real link."""
import socket
import unittest

from commonnet.net import client as client_mod
from commonnet.net import server as server_mod


class NoDelayTests(unittest.TestCase):
    def test_a_client_connection_has_nagle_turned_off(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        conn = client_mod._HTTPConnection("127.0.0.1", listener.getsockname()[1], timeout=5)
        self.addCleanup(conn.close)
        conn.connect()
        self.assertNotEqual(conn.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 0)   # 0 means Nagle is on

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
        self.assertNotEqual(handler.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 0)


if __name__ == "__main__":
    unittest.main()
