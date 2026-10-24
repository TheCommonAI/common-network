"""A TCP delay proxy in front of one node: emulates internet RTT, jitter and loss on loopback.

Each direction delays every chunk by rtt/2 (plus up to jitter/2 of random extra), keeping order. A new
connection also pays one extra RTT before its first byte reaches the node (the TCP handshake), so turning
keep-alive off costs what it would cost on a real path. Loss is modelled as TCP would show it: with probability
`loss` a chunk waits one retransmission timeout (RTO_MS) instead of being dropped. The settings can be changed
while the proxy runs.
"""
from __future__ import annotations

import queue
import random
import socket
import threading
import time

RTO_MS = 200.0
CHUNK = 65536
SPIN_S = 0.006


class DelayProxy:
    def __init__(self, upstream_port: int, *, port: int = 0, rtt_ms: float = 0.0, jitter_ms: float = 0.0,
                 loss: float = 0.0, seed: int = 0):
        self.upstream_port = upstream_port
        self.rtt_ms, self.jitter_ms, self.loss = rtt_ms, jitter_ms, loss
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", port))
        self._listener.listen(64)
        self.port = self._listener.getsockname()[1]
        self._stop = threading.Event()
        self._socks: set[socket.socket] = set()
        self._lock = threading.Lock()
        self.connections = 0
        threading.Thread(target=self._accept, daemon=True, name="delay-proxy").start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _delay_s(self) -> float:
        with self._rng_lock:
            extra = self._rng.uniform(0, self.jitter_ms) / 2 if self.jitter_ms else 0.0
            lost = self.loss > 0 and self._rng.random() < self.loss
        return (self.rtt_ms / 2 + extra + (RTO_MS if lost else 0.0)) / 1000.0

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _track(self, s: socket.socket) -> None:
        with self._lock:
            self._socks.add(s)

    def _serve(self, client: socket.socket) -> None:
        self._track(client)
        try:
            up = socket.create_connection(("127.0.0.1", self.upstream_port), timeout=5)
        except OSError:
            client.close()
            return
        self._track(up)
        for s in (client, up):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connections += 1
        handshake = self.rtt_ms / 1000.0
        threading.Thread(target=self._pump, args=(client, up, handshake), daemon=True).start()
        threading.Thread(target=self._pump, args=(up, client, 0.0), daemon=True).start()

    def _pump(self, src: socket.socket, dst: socket.socket, first_extra_s: float) -> None:
        q: queue.Queue = queue.Queue()

        def writer():
            while True:
                item = q.get()
                if item is None:
                    break
                due, data = item
                wait = due - time.monotonic()
                if wait > SPIN_S:
                    time.sleep(wait - SPIN_S)  # sleep() overshoots by 1-5 ms on macOS: spin the last bit
                while time.monotonic() < due:
                    pass
                try:
                    dst.sendall(data)
                except OSError:
                    break
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        w = threading.Thread(target=writer, daemon=True)
        w.start()
        last_due = 0.0
        try:
            while True:
                data = src.recv(CHUNK)
                if not data:
                    break
                due = max(last_due, time.monotonic() + self._delay_s() + first_extra_s)
                first_extra_s = 0.0
                last_due = due
                q.put((due, data))
        except OSError:
            pass
        q.put(None)
        w.join(timeout=30)
        for s in (src, dst):
            try:
                s.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        with self._lock:
            socks, self._socks = list(self._socks), set()
        for s in socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()
