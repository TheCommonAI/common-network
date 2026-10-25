"""Why a link is direct or tunnel: STUN ordering, per-hop reports, failure reasons, join-time summary."""
import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from commonnet.net import rtc
from commonnet.net.client import BlockClient, RemoteError, RtcPeers, Unreachable, _short

URL = "https://example.trycloudflare.com"
HEALTH = {"ok": True, "layers": [0, 4], "device": "stub"}


def make(**kw):
    logs = []
    return BlockClient(token=None, retries=0, rtc_log=logs.append, **kw), logs


class StunOrder(unittest.TestCase):
    def test_reachable_server_goes_first(self):
        urls = ("stun:a:1", "stun:b:2", "stun:c:3")
        ordered, ok = rtc.order_stun(urls, probe=lambda u: u == "stun:c:3", now=lambda: 1e9)
        self.assertEqual(ordered[0], "stun:c:3")
        self.assertTrue(ok)
        self.assertEqual(set(ordered), set(urls))

    def test_none_reachable_keeps_every_server_and_says_so(self):
        urls = ("stun:x:1", "stun:y:2")
        ordered, ok = rtc.order_stun(urls, probe=lambda u: False, now=lambda: 2e9)
        self.assertEqual(ordered, urls)
        self.assertFalse(ok)

    def test_more_than_two_defaults(self):
        self.assertGreaterEqual(len(rtc.DEFAULT_STUN_URLS), 3)
        self.assertLessEqual(len(rtc.DEFAULT_STUN_URLS), 4)


class JoinSummary(unittest.TestCase):
    def test_missing_dependency_is_named(self):
        with mock.patch.object(rtc, "available", return_value=False):
            line = rtc.join_summary(True)
        self.assertIn("UNAVAILABLE", line)
        self.assertIn("aiortc", line)

    def test_disabled(self):
        self.assertIn("off", rtc.join_summary(False))

    @unittest.skipUnless(rtc.available(), "aiortc unavailable")
    def test_no_stun_answers(self):
        with mock.patch.object(rtc, "order_stun", return_value=(rtc.DEFAULT_STUN_URLS, False)):
            self.assertIn("no STUN server answered", rtc.join_summary(True))


class Reasons(unittest.TestCase):
    def test_short_reasons_carry_no_secrets(self):
        self.assertIn("HTTP 503", _short(RemoteError(503, "Bearer abcdef secret sdp v=0")))
        self.assertNotIn("secret", _short(RemoteError(503, "Bearer abcdef secret")))
        self.assertEqual(_short(Unreachable("POST https://u:p@host/ failed")), "peer unreachable for signaling")
        self.assertEqual(_short(KeyError("token")), "KeyError")
        self.assertIn("ICE", _short(rtc.RtcUnavailable("ICE/DTLS did not connect within 30s")))


class Report(unittest.TestCase):
    def test_tunnel_with_old_peer_reason_and_log(self):
        client, logs = make()
        caps_less = dict(HEALTH)
        with mock.patch.object(rtc, "available", return_value=True), \
                mock.patch.object(client, "_http_call", return_value=caps_less):
            client.health(URL)
            client._rtc_states[URL]["future"].result(timeout=3)
        rep = client.transport_report(URL)
        self.assertEqual(rep["transport"], "tunnel")
        self.assertIn("does not offer direct links", rep["last_failure"])
        self.assertTrue(any("direct link failed" in line for line in logs))

    def test_failed_negotiation_is_recorded_and_logged(self):
        client, logs = make()

        class Peer:
            def __init__(self, *a, **k): pass
            def connect(self, timeout=None): raise rtc.RtcUnavailable("ICE/DTLS did not connect within 30s")
            def close(self): pass

        health = dict(HEALTH, caps=[rtc.CAPABILITY])
        with mock.patch.object(rtc, "available", return_value=True), \
                mock.patch.object(rtc, "RtcClient", Peer), \
                mock.patch.object(client, "_http_call", return_value=health):
            client.health(URL)
            client._rtc_states[URL]["future"].result(timeout=3)
        rep = client.transport_report(URL)
        self.assertEqual(rep, {"transport": "tunnel", "last_failure": "ICE/DTLS did not connect within 30s"})
        self.assertIn("example.trycloudflare.com", logs[-1])

    def test_direct_when_peer_ready(self):
        client, _ = make()
        client._rtc_states[URL] = {"peer": mock.Mock(ready=True), "future": None, "after": 0, "epoch": 0}
        self.assertEqual(client.transport_report(URL), {"transport": "direct (webrtc)", "last_failure": None})

    def test_disabled_and_missing_dependency_reasons(self):
        client, _ = make(rtc=False)
        self.assertIn("disabled", client.transport_report(URL)["last_failure"])
        client, _ = make(rtc=True)
        with mock.patch.object(rtc, "available", return_value=False):
            self.assertIn("aiortc is not installed", client.transport_report(URL)["last_failure"])

    def test_ip_endpoint_is_https_not_tunnel(self):
        client, _ = make(rtc=False)
        self.assertEqual(client.transport_report("https://203.0.113.5:8443")["transport"], "https")

    def test_shared_peers_survive_close_connections(self):
        peers = RtcPeers()
        peer = mock.Mock(ready=True)
        client, _ = make(rtc_peers=peers)
        peers[URL] = {"peer": peer, "future": None, "after": 0, "epoch": 0}
        client.close_connections()
        peer.close.assert_not_called()
        self.assertIn(URL, peers)
        peers.close()
        peer.close.assert_called_once()
        self.assertEqual(len(peers), 0)

    def test_unshared_peers_are_closed_with_the_client(self):
        client, _ = make()
        peer = mock.Mock(ready=True)
        client._rtc_states[URL] = {"peer": peer, "future": None, "after": 0, "epoch": 0}
        client.close_connections()
        peer.close.assert_called_once()

    def test_dropped_link_records_reason(self):
        client, logs = make()
        peer = mock.Mock(ready=True)
        client._rtc_states[URL] = {"peer": peer, "future": None, "after": 0, "epoch": 0}
        client._drop_rtc(URL, peer, "direct link failed mid-request (RtcUnavailable)")
        self.assertIn("mid-request", client.transport_report(URL)["last_failure"])
        self.assertTrue(logs)


class AskVerbose(unittest.TestCase):
    def test_transport_lines(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "common_cli_vis", Path(__file__).resolve().parents[2] / "common" / "common.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rows = [{"hop": "hop0", "layers": [0, 6], "transport": "direct (webrtc)", "last_failure": None},
                {"hop": "hop1", "layers": [6, 12], "transport": "tunnel", "last_failure": "UDP blocked"}]
        with mock.patch.object(mod, "_get_json", return_value={"transport": rows}):
            self.assertEqual(mod._transport_lines("http://x", False), [])
            lines = mod._transport_lines("http://x", True)
        self.assertEqual(lines[0], "transport layers 0-6: direct (webrtc)")
        self.assertEqual(lines[1], "transport layers 6-12: tunnel (direct link: UDP blocked)")


if __name__ == "__main__":
    unittest.main()
