"""Bounded signaling and real local ICE/DTLS/SCTP RPC, with no WAN claims."""
from __future__ import annotations

import concurrent.futures
import asyncio
import threading
import time
import unittest
from unittest import mock

from commonnet.net import rtc


def description(address="8.8.8.8", kind="offer", extra=""):
    sdp = ("v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n"
           "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n"
           "a=fingerprint:sha-256 " + ":".join(["AB"] * 32) + "\r\n"
           f"a=candidate:a 1 udp 1234 {address} 45678 typ host\r\n" + extra)
    return {"type": kind, "sdp": sdp}


class Signaling(unittest.TestCase):
    def test_public_literals_only_by_default(self):
        for address in ("8.8.8.8", "2606:4700:4700::1111"):
            with self.subTest(address=address):
                self.assertIn(address, rtc._description(description(address), "offer")["sdp"])
        for address in ("127.0.0.1", "192.168.20.41", "169.254.169.254", "100.64.0.1",
                        "0.0.0.0", "224.0.0.1", "example.com", "host.local", "::1", "fe80::1%en0",
                        "fec0::1", "::ffff:192.168.1.1", "::ffff:127.0.0.1"):
            with self.subTest(address=address), self.assertRaises(rtc.RtcUnavailable):
                rtc._description(description(address), "offer")
        self.assertIn("127.0.0.1", rtc._description(description("127.0.0.1"), "offer", True)["sdp"])

    def test_private_candidates_are_removed_without_removing_public_ones(self):
        value = description(extra="a=candidate:b 1 udp 2 192.168.1.1 55555 typ host\r\n")
        filtered = rtc._description(value, "offer")["sdp"]
        self.assertIn("8.8.8.8", filtered)
        self.assertNotIn("192.168.1.1", filtered)

    def test_media_fingerprints_and_sizes_are_validated(self):
        invalid = [dict(description(), unexpected=True), description(kind="answer"),
                   {"type": "offer", "sdp": "x" * (rtc.MAX_SDP_BYTES + 1)},
                   {"type": "offer", "sdp": description()["sdp"].replace("sha-256", "sha-1")},
                   {"type": "offer", "sdp": description()["sdp"] + "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"},
                   description(extra="a=candidate:b 1 udp 1 8.8.4.4 55555 typ host\r\n" * rtc.MAX_CANDIDATES)]
        for value in invalid:
            with self.subTest(value_type=value.get("type")), self.assertRaises(rtc.RtcProtocolError):
                rtc._description(value, "offer")

    def test_rpc_rejects_arbitrary_paths_and_nonfinite_messages(self):
        for method, path, body in (("POST", "/rtc/offer", {}), ("GET", "/manifest", None),
                                   ("POST", "http://169.254.169.254", {}), ("GET", "/health", {})):
            with self.subTest(path=path), self.assertRaises(rtc.RtcProtocolError):
                rtc._request(method, path, body)
        with self.assertRaises(rtc.RtcProtocolError):
            rtc._encode({"value": float("nan")})
        with self.assertRaises(rtc.RtcProtocolError):
            rtc._encode({"value": "x" * rtc.MAX_MESSAGE_BYTES})
        # Encoder recursion limits vary between supported Python builds.
        with mock.patch.object(rtc.json, "dumps", side_effect=RecursionError), self.assertRaises(rtc.RtcProtocolError):
            rtc._encode({"value": "bounded"})

    def test_malformed_and_expired_frames_release_the_reassembly_budget(self):
        class Channel:
            readyState = "open"
            bufferedAmount = 0
            def on(self, *args):
                pass
        class Peer:
            async def close(self):
                pass

        async def check():
            runtime = rtc._get_runtime()
            before = runtime.reassembly_bytes
            for malformed in (b"short", rtc._HEADER.pack(b"BAD!", 1, 1, 0) + b"a",
                              rtc._HEADER.pack(rtc._MAGIC, 1, rtc.MAX_MESSAGE_BYTES + 1, 0) + b"a",
                              rtc._HEADER.pack(rtc._MAGIC, 1, 2, 1) + b"a"):
                link = rtc._Link(Peer(), Channel())
                link.receive(malformed)
                self.assertTrue(link.closed)
                self.assertEqual(runtime.reassembly_bytes, before)
            link = rtc._Link(Peer(), Channel())
            link.receive(rtc._HEADER.pack(rtc._MAGIC, 1, 10, 0) + b"abc")
            self.assertEqual(runtime.reassembly_bytes, before + 3)
            link._expire(1)
            self.assertTrue(link.closed)
            self.assertEqual(link.assemblies, {})
            self.assertEqual(runtime.reassembly_bytes, before)
            nested = b"[" * 2000 + b"0" + b"]" * 2000
            link = rtc._Link(Peer(), Channel())
            link.receive(rtc._HEADER.pack(rtc._MAGIC, 2, len(nested), 0) + nested)
            self.assertTrue(link.closed)
            self.assertEqual(runtime.reassembly_bytes, before)
            # Duplicate queued frames after closure must be harmless.
            link.receive(b"short")
            await asyncio.sleep(0)
        rtc._get_runtime().run(check(), 3)


@unittest.skipUnless(rtc.available(), "optional aiortc is not installed")
class RealChannels(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.blocking = False
        self.started, self.release = threading.Event(), threading.Event()

        def handler(method, path, body):
            self.calls.append((method, path, body))
            if self.blocking:
                self.started.set()
                self.release.wait(2)
            return {"status": 200, "body": {"echo": body}, "server_ms": 0.125}

        self.server = rtc.RtcServer(handler, allow_private=True, stun_urls=[])
        self.client = rtc.RtcClient(self.server.offer, allow_private=True, stun_urls=[])
        self.addCleanup(self.server.close)
        self.addCleanup(self.client.close)
        self.addCleanup(self.release.set)
        self.client.connect(timeout=8)

    def test_reliable_large_messages_and_response_correlation(self):
        self.assertTrue(self.client.ready)
        payload = {"payload": "a" * (2 * 1024 * 1024), "value": 7}
        result = self.client.request("POST", "/forward", payload, timeout=15)
        self.assertEqual(result, {"status": 200, "body": {"echo": payload}, "server_ms": 0.125})
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self.client.request, "POST", "/session", {"id": str(i)}, 5) for i in range(3)]
            self.assertEqual([f.result()["body"]["echo"]["id"] for f in futures], ["0", "1", "2"])

    def test_status_and_handler_error_are_not_silently_success(self):
        self.server.handler = lambda *args: None
        # Existing connections keep the handler they authenticated with.
        self.client.close()
        self.client = rtc.RtcClient(self.server.offer, allow_private=True, stun_urls=[])
        self.addCleanup(self.client.close)
        self.client.connect(timeout=8)
        response = self.client.request("POST", "/session", {"id": "s"}, timeout=5)
        self.assertEqual(response["status"], 500)
        self.assertNotIn("sdp", str(response))

    def test_closed_channel_fails_without_leaving_pending_calls(self):
        self.server.close()
        deadline = time.monotonic() + 3
        while self.client.ready and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.client.ready)
        with self.assertRaises(rtc.RtcUnavailable):
            self.client.request("POST", "/session", {"id": "s"}, timeout=0.1)
        self.assertEqual(self.client._link.pending, {})

    def test_connection_limit_is_bounded(self):
        self.server.max_connections = 1
        another = rtc.RtcClient(self.server.offer, allow_private=True, stun_urls=[])
        self.addCleanup(another.close)
        with self.assertRaises(rtc.RtcUnavailable):
            another.connect(timeout=3)
        self.assertTrue(self.client.ready)

    def test_rpc_timeout_closes_link_and_clears_pending_reply(self):
        self.blocking = True
        started = time.monotonic()
        with self.assertRaises(rtc.RtcUnavailable):
            self.client.request("POST", "/forward", {"id": "s"}, timeout=0.05)
        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(self.client.ready)
        self.assertEqual(self.client._link.pending, {})
        self.release.set()

    def test_close_cancels_pending_rpc_without_waiting_for_model_work(self):
        self.blocking = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.client.request, "POST", "/forward", {"id": "s"}, 5)
            self.assertTrue(self.started.wait(2))
            self.client.close()
            with self.assertRaises(rtc.RtcUnavailable):
                future.result(timeout=1)
            self.assertEqual(self.client._link.pending, {})
            self.release.set()

    def test_canceled_negotiation_closes_the_partial_peer(self):
        gate = threading.Event()
        def stalled_signaler(offer):
            gate.wait(2)
            raise RuntimeError("signaling stopped")
        client = rtc.RtcClient(stalled_signaler, allow_private=True, stun_urls=[])
        self.addCleanup(client.close)
        try:
            with self.assertRaises(rtc.RtcUnavailable):
                client.connect(timeout=0.05)
            deadline = time.monotonic() + 1
            while client._pc is not None and client._pc.connectionState != "closed" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(client.ready)
            self.assertEqual(client._pc.connectionState, "closed")
        finally:
            gate.set()


if __name__ == "__main__":
    unittest.main()
