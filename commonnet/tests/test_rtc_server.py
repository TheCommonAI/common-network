"""Authenticated signaling and the shared HTTP / WebRTC block RPC contract."""
from __future__ import annotations

import json
import os
import struct
import sys
import types
import unittest
import urllib.error
import urllib.request
from unittest import mock

import commonnet.net as net_package
from commonnet.net import protocol
from commonnet.net.server import BlockServer, StubBlock

TOKEN = "rtc-server-test-credential-012345"


def post(server, path, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(server.url + path, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.load(exc)


def fake_core():
    module = types.ModuleType("commonnet.net.rtc")
    module.CAPABILITY = "rtc_v1"
    module.PRIVATE_CAPABILITY = "rtc_private_v1"
    module.available = lambda: True
    module.RtcProtocolError = type("RtcProtocolError", (ValueError,), {})
    module.RtcUnavailable = type("RtcUnavailable", (RuntimeError,), {})
    module.RtcServer = mock.Mock()
    return module


class ServerSignaling(unittest.TestCase):
    def server(self, **kw):
        server = BlockServer(StubBlock(0, 4), token=TOKEN, **kw).start()
        self.addCleanup(server.stop)
        return server

    def test_offer_requires_authentication_before_the_parser_or_handshake(self):
        server = self.server()
        extension = mock.Mock(return_value=(200, {"type": "answer", "sdp": "unauthenticated"}))
        server.extra_handlers[("POST", "/rtc/offer")] = extension
        with mock.patch.object(server, "_rtc_offer", return_value={"type": "answer", "sdp": "test"}) as offer:
            for token in (None, "incorrect-rtc-credential"):
                with self.subTest(authenticated=token is not None):
                    status, _ = post(server, "/rtc/offer", {"type": "offer", "sdp": "test"}, token)
                    self.assertEqual(status, 401)
            offer.assert_not_called()
            status, body = post(server, "/rtc/offer", {"type": "offer", "sdp": "test"}, TOKEN)
            self.assertEqual(status, 200)
            self.assertEqual(body["type"], "answer")
            offer.assert_called_once()
            extension.assert_not_called()

    def test_disabled_offer_is_501_and_does_not_advertise_rtc(self):
        server = self.server(rtc_enabled=False)
        status, body = post(server, "/rtc/offer", {"type": "offer", "sdp": "test"}, TOKEN)
        self.assertEqual(status, 501)
        self.assertNotIn("rtc_v1", server.caps)
        self.assertNotIn("sdp", body)

    def test_environment_can_disable_rtc(self):
        with mock.patch.dict(os.environ, {"COMMONNET_RTC": "0"}):
            server = self.server()
        self.assertFalse(server._rtc_available())
        self.assertNotIn("rtc_v1", server.caps)

    def test_available_capability_and_lazy_server_cleanup(self):
        core = fake_core()
        with mock.patch.dict(sys.modules, {"commonnet.net.rtc": core}), \
                mock.patch.object(net_package, "rtc", core, create=True):
            server = self.server(allow_chain_private=True, rtc_stun_urls=[])
            self.assertIn("rtc_v1", server.caps)
            self.assertIn("rtc_private_v1", server.caps)
            core.RtcServer.assert_not_called()
            endpoint = core.RtcServer.return_value
            endpoint.offer.return_value = {"type": "answer", "sdp": "test"}
            self.assertEqual(server._rtc_offer({"type": "offer", "sdp": "test"})["type"], "answer")
            self.assertEqual(core.RtcServer.call_args.kwargs,
                             {"allow_private": True, "stun_urls": [], "max_connections": 16})
            server.stop()
            endpoint.close.assert_called_once()
            self.assertNotIn("rtc_v1", server.caps)

    def test_offer_parser_errors_are_sanitized(self):
        core = fake_core()
        with mock.patch.dict(sys.modules, {"commonnet.net.rtc": core}), \
                mock.patch.object(net_package, "rtc", core, create=True):
            server = self.server()
            core.RtcServer.return_value.offer.side_effect = core.RtcProtocolError("private-signaling-details")
            status, body = post(server, "/rtc/offer", {"type": "offer", "sdp": "test"}, TOKEN)
            self.assertEqual(status, 400)
            self.assertNotIn("private-signaling-details", json.dumps(body))


class SharedRpc(unittest.TestCase):
    def setUp(self):
        self.block = StubBlock(0, 4)
        self.server = BlockServer(self.block, rtc_enabled=False)
        self.addCleanup(self.server.stop)
        self.hidden = protocol.Tensor((1, 2, 4), "float32", struct.pack("<8f", *range(8)))

    def test_open_forward_retry_truncate_and_close_share_the_http_session_table(self):
        call = self.server.rpc
        self.assertEqual(call("POST", "/session", {"id": "s"})["status"], 200)
        body = protocol.encode_forward_request("s", self.hidden, 0)
        first = call("POST", "/forward", body)
        retry = call("POST", "/forward", body)
        self.assertEqual(protocol.decode_forward_reply(first["body"]), self.hidden)
        self.assertEqual(first["body"], retry["body"])
        self.assertIsInstance(first["server_ms"], float)
        self.assertEqual(sum(c[0] == "forward" for c in self.block.calls), 1)
        self.assertEqual(call("POST", "/truncate", {"id": "s", "length": 1})["status"], 200)
        self.assertEqual(call("POST", "/close", {"id": "s"})["status"], 200)
        self.assertEqual(self.server._sessions, {})

    def test_status_mappings_and_rpc_recursion_are_bounded(self):
        cases = (("GET", "/health", None, 200), ("GET", "/forward", None, 405),
                 ("POST", "/health", {}, 405), ("POST", "/missing", {}, 404),
                 ("POST", "/rtc/offer", {}, 404), ("GET", "/manifest", None, 404),
                 ("POST", "/session", {"id": "not valid"}, 400),
                 ("POST", "/forward", protocol.encode_forward_request("missing", self.hidden, 0), 404))
        for method, path, body, status in cases:
            with self.subTest(method=method, path=path):
                response = self.server.rpc(method, path, body)
                self.assertEqual(response["status"], status)
                self.assertEqual(set(response), {"status", "body", "server_ms"})

    def test_public_extension_handlers_are_not_exposed_over_an_rtc_channel(self):
        handler = mock.Mock(return_value=(200, {"private": "unused"}))
        self.server.extra_handlers[("GET", "/manifest")] = handler
        self.assertEqual(self.server.rpc("GET", "/manifest")["status"], 404)
        handler.assert_not_called()

    def test_rpc_payload_limit_applies_before_block_execution(self):
        self.server.max_request_bytes = 32
        result = self.server.rpc("POST", "/session", {"id": "s", "padding": "x" * 100})
        self.assertEqual(result["status"], 413)
        self.assertEqual(self.block.calls, [])


def core_available():
    try:
        from commonnet.net import rtc
        return rtc.available()
    except ImportError:
        return False


@unittest.skipUnless(core_available(), "requires the WebRTC runtime")
class RealRtcBlock(unittest.TestCase):
    def test_datachannel_uses_authenticated_offer_and_shared_idempotent_rpc(self):
        from commonnet.net import rtc
        server = BlockServer(StubBlock(0, 4), token=TOKEN, allow_rtc_private=True, rtc_stun_urls=[]).start()
        self.addCleanup(server.stop)

        def signal(offer):
            status, answer = post(server, "/rtc/offer", offer, TOKEN)
            if status != 200:
                raise RuntimeError("signaling failed")
            return answer

        client = rtc.RtcClient(signal, allow_private=True, stun_urls=[])
        self.addCleanup(client.close)
        client.connect(timeout=15)
        self.assertTrue(client.ready)
        self.assertEqual(client.request("POST", "/session", {"id": "rtc"})["status"], 200)
        hidden = protocol.Tensor((1, 2, 4), "float32", struct.pack("<8f", *range(8)))
        body = protocol.encode_forward_request("rtc", hidden, 0)
        result = client.request("POST", "/forward", body)
        self.assertEqual(protocol.decode_forward_reply(result["body"]), hidden)
        self.assertEqual(client.request("POST", "/forward", body)["body"], result["body"])
        self.assertEqual(sum(c[0] == "forward" for c in server.block.calls), 1)
        self.assertEqual(client.request("POST", "/close", {"id": "rtc"})["status"], 200)
        self.assertEqual(server._sessions, {})

    def test_datachannel_chain_keeps_int8_forwarding_and_final_verification(self):
        from commonnet.net import rtc
        from commonnet.tests.test_chain_protocol import TokenBlock
        first = BlockServer(TokenBlock(0, 2, total=4), allow_chain_private=True,
                            rtc_stun_urls=[]).start()
        last = BlockServer(TokenBlock(2, 4, total=4), allow_chain_private=True,
                           rtc_stun_urls=[]).start()
        self.addCleanup(first.stop)
        self.addCleanup(last.stop)

        def signal(offer):
            status, answer = post(first, "/rtc/offer", offer)
            if status != 200:
                raise RuntimeError("signaling failed")
            return answer

        client = rtc.RtcClient(signal, allow_private=True, stun_urls=[])
        self.addCleanup(client.close)
        client.connect(timeout=15)
        route = [{"url": last.url, "layers": [2, 4], "token": None, "pin": None}]
        body = {"id": "rtc-chain", "ids": [4], "start_pos": 0, "layers": [0, 2],
                "route": route, "which": "last", "activation_codec": "int8_group_v1"}
        response = client.request("POST", "/chain", body)
        self.assertEqual((response["status"], response["body"]), (200, {"ids": [5]}))
        body.update(ids=[5, 6, 99], start_pos=1, which="all", drafts=[6, 99])
        response = client.request("POST", "/chain", body)
        self.assertEqual(response["status"], 200)
        self.assertEqual(response["body"], {"ids": [6, 7], "accepted": 1, "next": 7})
        self.assertEqual(client.request("POST", "/close", {"id": "rtc-chain"})["status"], 200)
        last.close("rtc-chain")


if __name__ == "__main__":
    unittest.main()
