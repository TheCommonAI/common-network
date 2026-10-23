"""Failover: a dead member is replaced by a replica, whose session is rebuilt by replaying what was sent.

Stub servers on 127.0.0.1 only. A killed process closes its sockets, so a test that stops a stub also
drops the transport's pooled connections to it.
"""
import unittest

import torch

from commonnet.client import BlockEndpoint
from commonnet.net.client import RemoteError, Unreachable
from commonnet.net.server import BlockServer, StubBlock
from commonnet.net.transport import HttpTransport

EP = BlockEndpoint("b", 4, 8)


def hidden(n=1, seed=0):
    return torch.randn(1, n, 32, generator=torch.Generator().manual_seed(seed))


class Failover(unittest.TestCase):
    def setUp(self):
        self.blocks = [StubBlock(4, 8), StubBlock(4, 8)]
        self.servers = [BlockServer(b, "127.0.0.1", 0).start() for b in self.blocks]
        self.addCleanup(lambda: [s.stop() for s in self.servers])

    def transport(self, replicas=True):
        return HttpTransport({"b": self.servers[0].url},
                             replicas={"b": [self.servers[1].url]} if replicas else None,
                             retries=0, sleep=lambda _: None)

    def kill(self, tr, index=0):
        self.servers[index].stop()
        for (name, i), client in tr._clients.items():
            if i == index:
                client.close_connections()

    def test_a_dead_member_is_replaced_and_its_session_replayed_in_order(self):
        tr = self.transport()
        first, second = hidden(5, 1), hidden(1, 2)
        self.assertTrue(torch.equal(tr.forward(EP, "s", first, 0), first))
        self.assertTrue(torch.equal(tr.forward(EP, "s", second, 5), second))
        self.kill(tr)
        third = hidden(1, 3)
        self.assertTrue(torch.equal(tr.forward(EP, "s", third, 6), third))
        self.assertEqual(tr.failovers, [("b", 0, 1)])
        replayed = [pos for op, _, pos in self.blocks[1].calls if op == "forward"]
        self.assertEqual(replayed, [0, 5, 6])  # the two earlier pieces, as sent, then the new one

    def test_later_calls_stay_on_the_replica(self):
        tr = self.transport()
        tr.forward(EP, "s", hidden(2), 0)
        self.kill(tr)
        tr.forward(EP, "s", hidden(1), 2)
        tr.forward(EP, "s", hidden(1), 3)
        self.assertEqual(len(tr.failovers), 1)

    def test_without_a_replica_a_dead_member_is_unreachable(self):
        tr = self.transport(replicas=False)
        tr.forward(EP, "s", hidden(2), 0)
        self.kill(tr)
        with self.assertRaises(Unreachable):
            tr.forward(EP, "s", hidden(1), 2)

    def test_when_every_server_is_dead_the_error_is_raised(self):
        tr = self.transport()
        tr.forward(EP, "s", hidden(2), 0)
        self.kill(tr, 0)
        self.servers[1].stop()
        with self.assertRaises(Unreachable):
            tr.forward(EP, "s", hidden(1), 2)

    def test_a_server_that_forgot_the_session_is_replaced_too(self):
        tr = self.transport()
        tr.forward(EP, "s", hidden(2), 0)
        self.servers[0]._sessions.pop("s")  # as if it restarted and forgot
        out = hidden(1, 9)
        self.assertTrue(torch.equal(tr.forward(EP, "s", out, 2), out))
        self.assertEqual(tr.failovers, [("b", 0, 1)])

    def test_other_errors_are_not_failed_over(self):
        tr = self.transport()
        def broken(session_id, hidden_tensor, start_pos):
            raise RuntimeError("out of memory")
        self.blocks[0].forward = broken
        with self.assertRaises(RemoteError) as caught:
            tr.forward(EP, "s", hidden(1), 0)  # a live server saying the block failed: 500
        self.assertEqual(caught.exception.status, 500)
        self.assertEqual(tr.failovers, [])

    def test_close_drops_the_replay_history(self):
        tr = self.transport()
        tr.forward(EP, "s", hidden(2), 0)
        tr.close(EP, "s")
        self.assertEqual(tr._history, {})


if __name__ == "__main__":
    unittest.main()
