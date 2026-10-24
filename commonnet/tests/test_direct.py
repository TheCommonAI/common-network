"""Direct mode, unit level: the address decision (all branches, with fakes), the signed fp/fpsig record fields and
how an old node sees them, the DHT dial-back, and the command-line wiring. No router and no Internet."""
from __future__ import annotations

import argparse
import base64
import json
import socket
import tempfile
import time
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from commonnet import direct
from commonnet.direct import DirectConfig, decide
from commonnet.mesh import records as rec
from commonnet.mesh.gossip import PeerTable
from commonnet.mesh.records import Identity
from commonnet.net import nat, secure

GLOBAL = "93.184.216.34"


class FakeMapping:
    def __init__(self, ip, port, method="upnp"):
        self.external_ip, self.external_port, self.method = ip, port, method
        self.released = False

    def release(self):
        self.released = True
        return True


def stun_ok(ip=GLOBAL):
    return lambda: {"public_ip": ip}


def stun_none():
    return lambda: {"public_ip": None}


def stun_raises():
    def f():
        raise OSError("no network")
    return f


def no_mapper(_port):
    raise nat.MappingFailed({"upnp": "no router"})


def always(answer):
    return lambda ip, port: answer


class DecideTests(unittest.TestCase):
    def test_loopback_test_mode_skips_every_check(self):
        d = decide(DirectConfig(loopback=True), 5000, always(False), mapper=lambda p: 1 / 0, stun=lambda: 1 / 0)
        self.assertEqual((d.kind, d.url), ("loopback", "https://127.0.0.1:5000"))

    def test_manual_forward_with_ip_and_port_verified(self):
        d = decide(DirectConfig(public_ip=GLOBAL, external_port=7000), 5000, always(True),
                   mapper=lambda p: 1 / 0, stun=lambda: 1 / 0)
        self.assertEqual((d.kind, d.url, d.verified), ("manual", f"https://{GLOBAL}:7000", True))

    def test_manual_port_only_takes_the_ip_from_stun(self):
        d = decide(DirectConfig(external_port=7000), 5000, always(True), mapper=lambda p: 1 / 0, stun=stun_ok())
        self.assertEqual(d.url, f"https://{GLOBAL}:7000")

    def test_manual_ip_only_keeps_the_local_port(self):
        d = decide(DirectConfig(public_ip=GLOBAL), 5000, always(True), mapper=lambda p: 1 / 0)
        self.assertEqual(d.url, f"https://{GLOBAL}:5000")

    def test_manual_with_nobody_to_ask_publishes_unverified(self):
        d = decide(DirectConfig(public_ip=GLOBAL, external_port=7000), 5000, always(None))
        self.assertTrue(d.direct)
        self.assertIsNone(d.verified)
        self.assertIn("unverified", d.detail)

    def test_manual_refused_by_the_dial_back_is_unreachable(self):
        d = decide(DirectConfig(public_ip=GLOBAL, external_port=7000), 5000, always(False))
        self.assertEqual(d.kind, "unreachable")
        self.assertIs(d.verified, False)

    def test_manual_without_any_ip_is_unreachable(self):
        d = decide(DirectConfig(external_port=7000), 5000, always(True), stun=stun_none())
        self.assertEqual(d.kind, "unreachable")
        self.assertIn("--public-ip", d.detail)

    def test_mapped_uses_the_routers_address_after_a_dial_back(self):
        m = FakeMapping(GLOBAL, 41234)
        d = decide(DirectConfig(), 5000, always(True), mapper=lambda p: m, stun=lambda: 1 / 0)
        self.assertEqual((d.kind, d.url), ("mapped", f"https://{GLOBAL}:41234"))
        self.assertFalse(m.released)

    def test_mapped_with_a_private_router_address_asks_stun_instead(self):
        m = FakeMapping("192.168.1.1", 41234)
        d = decide(DirectConfig(), 5000, always(True), mapper=lambda p: m, stun=stun_ok())
        self.assertEqual(d.url, f"https://{GLOBAL}:41234")

    def test_mapped_but_never_dialled_is_not_believed(self):
        m = FakeMapping(GLOBAL, 41234)
        d = decide(DirectConfig(), 5000, always(None), mapper=lambda p: m)
        self.assertEqual(d.kind, "unreachable")
        self.assertTrue(m.released)    # the mapping is deleted again

    def test_mapped_but_the_dial_back_fails_releases_the_mapping(self):
        m = FakeMapping(GLOBAL, 41234)
        d = decide(DirectConfig(), 5000, always(False), mapper=lambda p: m)
        self.assertEqual(d.kind, "unreachable")
        self.assertTrue(m.released)

    def test_cgnat_address_is_never_published(self):
        m = FakeMapping("100.64.3.4", 41234)
        d = decide(DirectConfig(), 5000, always(True), mapper=lambda p: m, stun=stun_ok("100.64.3.4"))
        self.assertEqual(d.kind, "unreachable")
        self.assertIsNone(d.url)
        self.assertTrue(m.released)

    def test_stun_address_with_the_local_port_when_there_is_no_mapping(self):
        d = decide(DirectConfig(), 5000, always(True), mapper=no_mapper, stun=stun_ok())
        self.assertEqual((d.kind, d.url), ("stun", f"https://{GLOBAL}:5000"))

    def test_stun_address_not_reachable_falls_back(self):
        d = decide(DirectConfig(), 5000, always(False), mapper=no_mapper, stun=stun_ok())
        self.assertEqual(d.kind, "unreachable")
        self.assertEqual(d.port, 5000)

    def test_a_private_stun_answer_is_refused(self):
        d = decide(DirectConfig(), 5000, always(True), mapper=no_mapper, stun=stun_ok("10.0.0.5"))
        self.assertEqual(d.kind, "unreachable")

    def test_no_stun_and_no_mapping_is_unreachable(self):
        for stun in (stun_none(), stun_raises()):
            self.assertEqual(decide(DirectConfig(), 5000, always(True), mapper=no_mapper, stun=stun).kind, "unreachable")

    def test_a_mapper_crash_is_not_fatal(self):
        d = decide(DirectConfig(), 5000, always(True), mapper=lambda p: 1 / 0, stun=stun_ok())
        self.assertEqual(d.kind, "stun")

    def test_dial_back_helper_that_raises_counts_as_not_asked(self):
        def boom(ip, port):
            raise OSError("x")
        self.assertEqual(decide(DirectConfig(), 5000, boom, mapper=no_mapper, stun=stun_ok()).kind, "unreachable")

    def test_forward_hint_names_the_ports(self):
        text = direct.forward_hint(7471, 7472)
        self.assertIn("TCP 7471", text)
        self.assertIn("UDP 7472", text)


class FallbackTests(unittest.TestCase):
    """Node._serve_direct with a fake decider: unreachable falls back to the tunnel path, or refuses."""

    def node(self, tmp, cfg, decider):
        from commonnet.bench.synth import SynthBlock, manifest_dict
        from commonnet.mesh.manifest import Manifest
        from commonnet.node import Hardware, Node
        node = Node(Manifest.load(manifest_dict(1, 2, network_id="t-direct-unit")), [], Identity.generate(),
                    Hardware("synth", 2.0, "synthetic"), port=0, tunnel=True, direct=cfg, state_dir=tmp,
                    decider=decider, log=lambda m: None)
        return node, SynthBlock(0, 2)

    def test_unreachable_with_fallback_returns_false_and_frees_the_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            node, block = self.node(tmp, DirectConfig(loopback=True),
                                    lambda *a, **k: direct.Decision("unreachable", detail="blocked"))
            self.assertFalse(node._serve_direct(block))
            self.assertIsNone(node.server)
            self.assertIsNone(node.fp)

    def test_unreachable_without_fallback_refuses_and_names_the_port(self):
        from commonnet.node import NodeError
        with tempfile.TemporaryDirectory() as tmp:
            node, block = self.node(tmp, DirectConfig(loopback=True, tunnel_fallback=False, dht_port=7472),
                                    lambda *a, **k: direct.Decision("unreachable", detail="blocked"))
            with self.assertRaises(NodeError) as cm:
                node._serve_direct(block)
            text = str(cm.exception)
            self.assertIn("forward TCP", text)
            self.assertIn("UDP 7472", text)
            self.assertIn(str(node.port), text)

    def test_direct_serves_tls_and_signs_the_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            node, block = self.node(tmp, DirectConfig(loopback=True),
                                    lambda cfg, port, dial, **k: direct.Decision("loopback", "127.0.0.1", port))
            self.assertTrue(node._serve_direct(block))
            try:
                r = node.record()
                self.assertTrue(rec.verify_record(r))
                self.assertEqual(r["url"], f"https://127.0.0.1:{node.server.port}")
                self.assertTrue(secure.verify_binding(r["pub"], r["fp"], r["fpsig"]))
            finally:
                node._teardown_server()


class RecordFieldTests(unittest.TestCase):
    def setUp(self):
        self.ident = Identity.generate()
        self.fp = "ab" * 32
        self.base = dict(network="n", url="https://93.184.216.34:7000", token="t", window=[0, 2], backend="cpu",
                         state="up", gib=1.0)

    def test_signed_with_a_binding_by_default(self):
        r = self.ident.sign_record(dict(self.base, fp=self.fp))
        self.assertTrue(rec.verify_record(r))
        self.assertEqual(r["fp"], self.fp)
        self.assertTrue(secure.verify_binding(r["pub"], r["fp"], r["fpsig"]))

    def test_tunnel_and_plain_records_are_exactly_what_they_were(self):
        r = self.ident.sign_record(dict(self.base, url="https://x.trycloudflare.com"))
        self.assertEqual(set(r), set(rec.SIGNED_FIELDS) | {"sig"})
        self.assertTrue(rec.verify_record(r))
        self.assertTrue(rec.verify_record(self.ident.sign_record(dict(self.base, url="http://127.0.0.1:1"))))

    def test_direct_url_without_fp_is_refused(self):
        self.assertFalse(rec.verify_record(self.ident.sign_record(self.base)))

    def test_binding_by_another_key_is_refused(self):
        evil = Identity.generate()
        r = self.ident.sign_record(dict(self.base, fp=self.fp, fpsig=secure.bind_fingerprint(evil, self.fp)))
        self.assertFalse(rec.verify_record(r))

    def test_tampering_with_fp_breaks_the_record_signature(self):
        r = self.ident.sign_record(dict(self.base, fp=self.fp))
        r["fp"] = "cd" * 32
        self.assertFalse(rec.verify_record(r))

    def test_fp_without_fpsig_or_malformed_is_refused(self):
        r = self.ident.sign_record(dict(self.base, fp=self.fp))
        del r["fpsig"]
        self.assertFalse(rec.verify_record(r))
        with self.assertRaises(ValueError):
            self.ident.sign_record(dict(self.base, fp="zz"))

    def test_an_old_node_still_verifies_and_keeps_a_new_record(self):
        """What a node that predates fp/fpsig does: the required keys are there and the signature covers every key
        but `sig`; its gossip table stores the record whole and hands the extra fields on."""
        now = time.time()
        r = self.ident.sign_record(dict(self.base, fp=self.fp), now=now)
        for key in rec.SIGNED_FIELDS:
            self.assertIn(key, r)
        body = {k: v for k, v in r.items() if k != "sig"}
        canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        Ed25519PublicKey.from_public_bytes(base64.b64decode(r["pub"])).verify(base64.b64decode(r["sig"]), canon)
        table = PeerTable("n")
        self.assertTrue(table.merge(r))
        self.assertEqual(table.get(r["peer"])["fp"], self.fp)

    def test_route_kind(self):
        self.assertEqual(rec.route_kind("https://1.2.3.4:5"), "direct")
        self.assertEqual(rec.route_kind("https://a.trycloudflare.com"), "tunnel")
        self.assertEqual(rec.route_kind("http://127.0.0.1:5"), "plain")

    def test_a_worst_case_record_fits_the_dht_reply_budget(self):
        from commonnet.dht.node import MAX_PACKET, REQ_PAD, _encode
        r = self.ident.sign_record(dict(network="common-" + "x" * 30, url="https://255.255.255.255:65535",
                                        token="t" * 32, window=[100, 200], backend="mlx", state="loading", gib=999.99,
                                        fp=self.fp))
        reply = {"t": "vals", "n": "0" * 16, "id": self.ident.peer, "pub": self.ident.pub_b64, "tok": "t" * 24,
                 "sig": "A" * 88, "more": True, "vals": [r], "nodes": [[self.ident.peer, "255.255.255.255", 65535]] * 3}
        self.assertLess(len(_encode(reply)), REQ_PAD)
        self.assertLessEqual(REQ_PAD, MAX_PACKET)


class DialBackTests(unittest.TestCase):
    def setUp(self):
        from commonnet.dht import DhtNode
        self.nodes = [DhtNode(Identity.generate(), bind=("127.0.0.1", 0), network_id="n", allow_private_dial=True)
                      for _ in range(3)]
        for n in self.nodes:
            n.start()
            self.addCleanup(n.stop)
        a, b, c = self.nodes
        a.bootstrap([b.address])
        c.bootstrap([b.address])
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.addCleanup(self.listener.close)

    def test_a_helper_connects_back_to_the_requesters_open_port(self):
        self.assertTrue(self.nodes[0].dial_back("127.0.0.1", self.listener.getsockname()[1]))

    def test_a_closed_port_answers_false(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.assertIs(self.nodes[0].dial_back("127.0.0.1", port), False)

    def test_the_helper_dials_only_the_requesters_own_address(self):
        self.assertIs(self.nodes[0].dial_back("203.0.113.77", self.listener.getsockname()[1]), False)

    def test_no_contacts_means_nobody_could_be_asked(self):
        from commonnet.dht import DhtNode
        lone = DhtNode(Identity.generate(), bind=("127.0.0.1", 0), network_id="n")
        lone.start()
        self.addCleanup(lone.stop)
        self.assertIsNone(lone.dial_back("127.0.0.1", 1))

    def test_a_default_helper_refuses_private_targets(self):
        from commonnet.dht import DhtNode
        strict = DhtNode(Identity.generate(), bind=("127.0.0.1", 0), network_id="n")
        strict.start()
        self.addCleanup(strict.stop)
        a = self.nodes[0]
        for other in self.nodes[1:]:
            a.table.remove(other.id)
        a.bootstrap([strict.address])
        self.assertIs(a.dial_back("127.0.0.1", self.listener.getsockname()[1]), False)

    def test_an_unsigned_request_is_not_answered(self):
        a, b = self.nodes[0], self.nodes[1]
        sent_before = b.stats["sent"]
        a._sendto(json.dumps({"t": "dial", "n": "0" * 16, "id": a.peer, "ip": "127.0.0.1", "port": 1,
                              "tok": "x"}).encode(), b.address)
        time.sleep(0.4)
        self.assertEqual(b.stats["sent"], sent_before)


class CliTests(unittest.TestCase):
    def parse(self, *argv):
        from commonnet import node
        return node.build_parser().parse_args(list(argv))

    def test_direct_off_by_default(self):
        from commonnet import node
        self.assertIsNone(node.direct_config(self.parse("--stub")))

    def test_direct_options_without_direct_are_an_error(self):
        from commonnet import node
        with self.assertRaises(node.NodeError):
            node.direct_config(self.parse("--bootstrap", "1.2.3.4:5"))

    def test_direct_config(self):
        from commonnet import node
        cfg = node.direct_config(self.parse("--direct", "--bootstrap", "1.2.3.4:5", "--public-ip", GLOBAL,
                                            "--external-port", "7000", "--no-tunnel-fallback"))
        self.assertEqual((cfg.bootstrap, cfg.public_ip, cfg.external_port, cfg.tunnel_fallback),
                         (["1.2.3.4:5"], GLOBAL, 7000, False))

    def test_private_public_ip_and_no_tunnel_are_refused(self):
        from commonnet import node
        for argv in (("--direct", "--public-ip", "192.168.1.5"), ("--direct", "--no-tunnel")):
            with self.assertRaises(node.NodeError):
                node.direct_config(self.parse(*argv))

    def test_network_dht_means_the_built_in_manifest_and_no_seed(self):
        from commonnet import node
        manifest, seeds = node.resolve_network("dht", [], environ={})
        self.assertEqual(seeds, [])
        self.assertEqual(manifest.id, "common-main-8b-v1")


if __name__ == "__main__":
    unittest.main()
