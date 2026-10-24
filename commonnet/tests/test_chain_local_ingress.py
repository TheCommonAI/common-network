"""Local client shortcuts must never become another machine's forwarding URL."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from commonnet.client import BlockEndpoint
from commonnet.node import Node
from commonnet.net.transport import HttpTransport


class LocalChainIngress(unittest.TestCase):
    def rig(self, own_index):
        urls = {"first": "https://8.8.8.8:7471", "last": "https://9.9.9.9:7471"}
        pins = {"8.8.8.8:7471": "a" * 64, "9.9.9.9:7471": "b" * 64}
        endpoints = [BlockEndpoint("first", 0, 18), BlockEndpoint("last", 18, 36)]
        token_names = {name: "synthetic-stage-credential-" + name for name in urls}
        health_calls, chain_calls = [], []
        def client_factory(**kw):
            name = next(n for n, token in token_names.items() if token == kw["token"])
            ep = next(e for e in endpoints if e.name == name)
            client = SimpleNamespace(token=kw["token"], _pin=kw.get("pin"))
            def health(url):
                health_calls.append((name, url))
                return ((ep.start, ep.end), "test", ("chain_v1", "activation_int8_group_v1",
                                                     "ids_in" if ep.start == 0 else "head_out"))
            def chain(url, body):
                chain_calls.append((name, url, body))
                return {"ids": [5]}
            client.health_info, client.chain = health, chain
            return client
        patch = mock.patch("commonnet.net.transport.BlockClient", side_effect=client_factory)
        patch.start()
        self.addCleanup(patch.stop)
        transport = HttpTransport(urls, tokens=token_names, pin=pins, retries=0)
        own = list(urls.values())[own_index]
        fp = list(pins.values())[own_index]
        node = SimpleNamespace(_lock=threading.RLock(), url=own, fp=fp,
                               server=SimpleNamespace(url="https://0.0.0.0:19012", port=19012))
        Node.localize(node, transport)
        return endpoints, transport, health_calls, chain_calls, own, fp

    def test_own_final_window_stays_public_in_forwarded_chain(self):
        eps, tr, health, calls, own, fp = self.rig(1)
        self.assertTrue(tr.chain_route(eps))
        self.assertEqual(tr.urls["last"], own)
        self.assertIn(("last", "https://127.0.0.1:19012"), health)
        tr.chain_step(eps, "s", [3, 4], 0)
        _, ingress, body = calls[-1]
        self.assertEqual(ingress, tr.urls["first"])
        self.assertEqual(body["route"][0]["url"], own)
        self.assertEqual(body["route"][0]["pin"], fp)
        self.assertEqual(body["activation_codec"], "int8_group_v1")
        self.assertEqual(tr._kw["pin"]["127.0.0.1:19012"], fp)

    def test_own_first_window_uses_local_ingress_but_keeps_public_identity(self):
        eps, tr, health, calls, own, fp = self.rig(0)
        self.assertTrue(tr.chain_route(eps))
        tr.chain_step(eps, "s", [3, 4], 0)
        _, ingress, body = calls[-1]
        self.assertEqual(ingress, "https://127.0.0.1:19012")
        self.assertEqual(tr.urls["first"], own)
        self.assertEqual(body["route"][0]["url"], tr.urls["last"])
        self.assertEqual(tr._chain_sessions["s"]["hops"][0]["pin"], fp)

    def test_replica_refresh_deduplicates_public_identity_and_keeps_local_shortcut(self):
        eps, tr, _, _, own, _ = self.rig(1)
        original = tr._candidates("last")
        self.assertEqual(tr._merge_servers("last", [(own, tr._tokens["last"])]), 0)
        self.assertEqual(tr._candidates("last"), original)
        self.assertEqual(tr._target("last", 0)[1], "https://127.0.0.1:19012")
        # The same canonical public URL is localized when selected as an alternate.
        tr._replicas["first"] = [(own, tr._tokens["last"])]
        self.assertEqual(tr._target("first", 1)[1], "https://127.0.0.1:19012")


if __name__ == "__main__":
    unittest.main()
