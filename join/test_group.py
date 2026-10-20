"""Offline tests for distributed groups: tokens, the WebSocket link, enrolment, and the GPU gate.

The end-to-end test runs the real relay on 127.0.0.1 and a real member agent. The
tunnel is a stub that returns the local address, so nothing calls Cloudflare. The
RPC server is a small Python stand-in for ggml-rpc-server that lists one CUDA
device and refuses to serve unless it is started on that device. It never starts
llama.cpp or a model.
"""
import hashlib
import hmac
import io
import json
import os
import secrets
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import group

TUNNEL_URL = "https://lab-test.trycloudflare.com"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def forge(secret: bytes, payload: dict) -> str:
    """A token signed as the coordinator would sign it, for any payload the test wants."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(secret, body, hashlib.sha256).digest()
    return f"{group.TOKEN_PREFIX}{group._b64e(body)}.{group._b64e(signature)}"


def make_state(directory: Path, tunnel_url: str | None = TUNNEL_URL) -> group.GroupState:
    """A coordinator record without a relay process. Enrolment needs no certificate and no socket."""
    data = {"v": 2, "group": "lab", "tunnel_url": tunnel_url,
            "secret": secrets.token_hex(32), "members": {}, "used": {}}
    state = group.GroupState(directory / "group.json", data)
    state.save()
    return state


class TokenTests(unittest.TestCase):
    SECRET = bytes(range(32))
    NOW = 1_800_000_000.0

    def token(self, **overrides):
        fields = dict(group="lab", member="node-a", tunnel_url=TUNNEL_URL, expires=self.NOW + 3600)
        fields.update(overrides)
        return group.issue_token(self.SECRET, **fields)

    def test_valid_token_round_trips_its_fields(self):
        payload = group.check_token(self.SECRET, self.token(), self.NOW)
        self.assertEqual((payload["g"], payload["m"], payload["u"]), ("lab", "node-a", TUNNEL_URL))
        self.assertNotIn("f", payload)  # no certificate fingerprint any more

    def test_token_carries_a_one_time_secret_and_the_expiry(self):
        payload = group.read_token(self.token())[0]
        self.assertRegex(payload["n"], r"[0-9a-f]{32}")
        self.assertEqual(payload["x"], int(self.NOW + 3600))

    def test_each_token_gets_its_own_one_time_secret(self):
        first = group.read_token(self.token())[0]["n"]
        second = group.read_token(self.token())[0]["n"]
        self.assertNotEqual(first, second)

    def test_wrong_secret_is_rejected(self):
        with self.assertRaisesRegex(group.GroupError, "not valid for this group"):
            group.check_token(bytes(32), self.token(), self.NOW)

    def test_tampered_tunnel_address_is_rejected(self):
        token = self.token()
        prefix, body, signature = token.split(".")
        forged_body = group._b64e(json.dumps({**json.loads(group._b64d(body)), "u": "https://evil.example"},
                                             sort_keys=True, separators=(",", ":")).encode())
        with self.assertRaisesRegex(group.GroupError, "not valid for this group"):
            group.check_token(self.SECRET, f"{prefix}.{forged_body}.{signature}", self.NOW)

    def test_expired_token_is_rejected_at_the_expiry_instant(self):
        token = self.token(expires=self.NOW)
        with self.assertRaisesRegex(group.GroupError, "expired"):
            group.check_token(self.SECRET, token, self.NOW)

    def test_malformed_tokens_are_refused_with_a_message(self):
        good = self.token()
        for bad in ("", "nope", "cng1.onlyone", "cng1.a.b.c", "cng1.!!!.sig", good[:-6] + "=====",
                    "cng1." + group._b64e(b"[1,2]") + ".sig", None):
            with self.subTest(token=bad):
                with self.assertRaises(group.GroupError):
                    group.check_token(self.SECRET, bad, self.NOW)

    def test_only_version_2_is_accepted(self):
        base = {"g": "lab", "m": "node-a", "u": TUNNEL_URL, "x": int(self.NOW) + 60, "n": "a" * 32}
        with self.assertRaisesRegex(group.GroupError, "unsupported group token version"):
            group.read_token(forge(self.SECRET, {**base, "v": 1}))
        self.assertEqual(group.read_token(forge(self.SECRET, {**base, "v": 2}))[0]["u"], TUNNEL_URL)

    def test_a_token_naming_a_plain_http_remote_tunnel_is_refused(self):
        base = {"v": 2, "g": "lab", "m": "node-a", "x": int(self.NOW) + 60, "n": "a" * 32}
        with self.assertRaisesRegex(group.GroupError, "must use https"):
            group.read_token(forge(self.SECRET, {**base, "u": "http://192.0.2.10:7443"}))

    def test_bad_tunnel_addresses_are_refused_when_issuing(self):
        for bad in ("", "lab.trycloudflare.com", "https://lab.example/path", "https://lab.example?x=1",
                    "https://user@lab.example", "https://", "ftp://lab.example", "https://lab.example:99999",
                    "http://192.0.2.10:7443"):
            with self.subTest(url=bad):
                with self.assertRaises(group.GroupError):
                    self.token(tunnel_url=bad)

    def test_a_trailing_slash_is_dropped_from_the_tunnel_address(self):
        self.assertEqual(group.read_token(self.token(tunnel_url=TUNNEL_URL + "/"))[0]["u"], TUNNEL_URL)

    def test_names_are_validated_when_issuing(self):
        with self.assertRaises(group.GroupError):
            self.token(member="Not Valid!")
        with self.assertRaises(group.GroupError):
            self.token(group="")


class TunnelAddressTests(unittest.TestCase):
    def test_https_and_loopback_http_are_accepted(self):
        self.assertEqual(group._check_tunnel_url("https://lab.trycloudflare.com/"), "https://lab.trycloudflare.com")
        self.assertEqual(group._check_tunnel_url("http://127.0.0.1:8080"), "http://127.0.0.1:8080")
        self.assertEqual(group._check_tunnel_url("http://localhost:8080"), "http://localhost:8080")

    def test_plain_http_to_a_remote_host_is_refused(self):
        with self.assertRaisesRegex(group.GroupError, "must use https"):
            group._check_tunnel_url("http://192.0.2.10:7443")


class FramingTests(unittest.TestCase):
    def test_round_trip_and_split_input(self):
        frames = [group.encode_frame(group.OPEN, 1), group.encode_frame(group.DATA, 1, b"hello"),
                  group.encode_frame(group.PING, 0), group.encode_frame(group.CLOSE, 1)]
        stream = b"".join(frames)
        decoder = group.FrameDecoder()
        decoded = []
        for i in range(len(stream)):  # one byte at a time
            decoded += decoder.feed(stream[i:i + 1])
        self.assertEqual(decoded, [(group.OPEN, 1, b""), (group.DATA, 1, b"hello"),
                                   (group.PING, 0, b""), (group.CLOSE, 1, b"")])

    def test_several_frames_in_one_read(self):
        data = group.encode_frame(group.DATA, 2, b"a") + group.encode_frame(group.DATA, 3, b"bb")
        self.assertEqual(group.FrameDecoder().feed(data), [(group.DATA, 2, b"a"), (group.DATA, 3, b"bb")])

    def test_partial_frame_waits_for_the_rest(self):
        decoder = group.FrameDecoder()
        frame = group.encode_frame(group.DATA, 4, b"payload")
        self.assertEqual(decoder.feed(frame[:-3]), [])
        self.assertEqual(decoder.feed(frame[-3:]), [(group.DATA, 4, b"payload")])

    def test_oversized_header_is_refused_before_the_payload_arrives(self):
        header = group.FRAME_HEADER.pack(group.DATA, 1, group.MAX_FRAME_PAYLOAD + 1)
        with self.assertRaisesRegex(group.FrameError, "over the"):
            group.FrameDecoder().feed(header)

    def test_unknown_kind_and_reserved_streams_are_refused(self):
        cases = [group.FRAME_HEADER.pack(9, 1, 0), group.FRAME_HEADER.pack(group.DATA, 0, 1),
                 group.FRAME_HEADER.pack(group.PING, 3, 0), group.FRAME_HEADER.pack(group.OPEN, 1, 4)]
        for raw in cases:
            with self.subTest(header=raw.hex()):
                with self.assertRaises(group.FrameError):
                    group.FrameDecoder().feed(raw)

    def test_encoder_applies_the_same_rules(self):
        with self.assertRaises(group.FrameError):
            group.encode_frame(group.DATA, 0, b"x")
        with self.assertRaises(group.FrameError):
            group.encode_frame(group.DATA, 2**32, b"x")
        with self.assertRaises(group.FrameError):
            group.encode_frame(group.DATA, 1, b"x" * (group.MAX_FRAME_PAYLOAD + 1))


class _EchoServer:
    """An in-process stand-in for a ggml-rpc-server: echoes every byte back."""

    def __init__(self, case):
        self.listener = socket.create_server(("127.0.0.1", 0))
        case.addCleanup(self.listener.close)
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._echo, args=(conn,), daemon=True).start()

    @staticmethod
    def _echo(conn):
        with conn:
            while True:
                data = conn.recv(65536)
                if not data:
                    return
                conn.sendall(data)


class MuxTests(unittest.TestCase):
    """The coordinator's and the member's ends of one link, over a plain socket pair."""

    def setUp(self):
        self.echo = _EchoServer(self)
        coordinator_sock, member_sock = socket.socketpair()
        self.coordinator = group.Mux(coordinator_sock)
        self.member = group.Mux(member_sock,
                                on_open=lambda _stream: group._connect_rpc(self.echo.port))
        for mux in (self.coordinator, self.member):
            threading.Thread(target=mux.run, daemon=True).start()
            self.addCleanup(mux.close)

    def test_bytes_round_trip_through_a_stream(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        self.assertIsNotNone(self.coordinator.open_stream(local))
        payload = b"x" * 300_000  # several DATA frames
        peer.settimeout(10)
        threading.Thread(target=peer.sendall, args=(payload,), daemon=True).start()
        received = bytearray()
        while len(received) < len(payload):
            chunk = peer.recv(65536)
            self.assertTrue(chunk, "stream closed early")
            received += chunk
        self.assertEqual(bytes(received), payload)

    def test_closing_the_local_end_closes_the_rpc_connection(self):
        local, peer = socket.socketpair()
        self.addCleanup(peer.close)
        self.coordinator.open_stream(local)
        peer.sendall(b"ping")
        peer.settimeout(5)
        self.assertEqual(peer.recv(4), b"ping")
        peer.close()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and self.member._streams:
            time.sleep(0.05)
        self.assertEqual(self.member._streams, {})
        self.assertEqual(self.coordinator._streams, {})

    def test_refused_open_closes_the_local_connection(self):
        self.member.on_open = lambda _stream: None
        local, peer = socket.socketpair()
        self.addCleanup(peer.close)
        self.coordinator.open_stream(local)
        peer.settimeout(5)
        self.assertEqual(peer.recv(16), b"")

    def test_open_on_a_closed_link_is_refused(self):
        self.coordinator.close()
        local, peer = socket.socketpair()
        self.addCleanup(peer.close)
        self.addCleanup(local.close)
        self.assertIsNone(self.coordinator.open_stream(local))


class WebSocketTests(unittest.TestCase):
    def pair(self):
        a, b = socket.socketpair()
        a.settimeout(5)
        b.settimeout(5)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        return a, b

    def test_accept_key_matches_the_rfc_example(self):
        self.assertEqual(group._accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_handshake_then_bytes_both_ways(self):
        client_sock, server_sock = self.pair()
        result = {}

        def server():
            _, headers = group._read_http_head(server_sock)
            link = group.server_upgrade(server_sock, headers)
            result["got"] = link.recv(1024)
            link.sendall(b"pong from server")

        threading.Thread(target=server, daemon=True).start()
        client = group.client_upgrade(client_sock, "lab-test.example", group.LINK_PATH)
        client.sendall(b"ping from client")
        self.assertEqual(client.recv(1024), b"pong from server")
        deadline = time.monotonic() + 5
        while "got" not in result and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(result["got"], b"ping from client")

    def test_a_large_payload_crosses_in_both_directions(self):
        client_sock, server_sock = self.pair()
        payload = os.urandom(300_000)

        def server():
            _, headers = group._read_http_head(server_sock)
            link = group.server_upgrade(server_sock, headers)
            received = bytearray()
            while len(received) < len(payload):
                received += link.recv(65536)
            link.sendall(bytes(received))

        threading.Thread(target=server, daemon=True).start()
        client = group.client_upgrade(client_sock, "lab-test.example", group.LINK_PATH)
        threading.Thread(target=client.sendall, args=(payload,), daemon=True).start()
        echoed = bytearray()
        while len(echoed) < len(payload):
            echoed += client.recv(65536)
        self.assertEqual(bytes(echoed), payload)

    def test_a_client_that_is_not_upgraded_gets_no_link(self):
        client_sock, server_sock = self.pair()

        def server():
            _http_only_reply(server_sock)

        threading.Thread(target=server, daemon=True).start()
        with self.assertRaisesRegex(group.GroupError, "did not accept the group link"):
            group.client_upgrade(client_sock, "lab-test.example", group.LINK_PATH)

    def test_server_refuses_an_unmasked_client_frame(self):
        client_sock, server_sock = self.pair()
        server = group.WebSocketStream(server_sock, client=False)
        # A server-side frame looks like a client's frame with the mask bit clear.
        client_sock.sendall(group.WebSocketStream(client_sock, client=False)._frame(group.WS_OP_BINARY, b"hi"))
        with self.assertRaisesRegex(group.GroupError, "masking"):
            server.recv(10)

    def test_client_refuses_a_masked_server_frame(self):
        client_sock, server_sock = self.pair()
        client = group.WebSocketStream(client_sock, client=True)
        server_sock.sendall(group.WebSocketStream(server_sock, client=True)._frame(group.WS_OP_BINARY, b"hi"))
        with self.assertRaisesRegex(group.GroupError, "masking"):
            client.recv(10)

    def test_ping_is_answered_with_a_pong_and_the_data_still_arrives(self):
        client_sock, server_sock = self.pair()
        client = group.WebSocketStream(client_sock, client=True)
        peer = group.WebSocketStream(server_sock, client=False)
        peer.sock.sendall(peer._frame(group.WS_OP_PING, b"are you there"))
        peer.sock.sendall(peer._frame(group.WS_OP_BINARY, b"hi"))
        self.assertEqual(client.recv(2), b"hi")
        self.assertEqual(bytes(client._out)[:1], bytes([0x80 | group.WS_OP_PONG]))

    def test_oversized_frame_is_refused_before_its_payload_arrives(self):
        client_sock, server_sock = self.pair()
        client = group.WebSocketStream(client_sock, client=True)
        server_sock.sendall(bytes([0x82, 127]) + struct.pack("!Q", group.WS_MAX_PAYLOAD + 1))
        with self.assertRaisesRegex(group.GroupError, "over the"):
            client.recv(10)

    def test_a_reserved_opcode_is_refused(self):
        client_sock, server_sock = self.pair()
        client = group.WebSocketStream(client_sock, client=True)
        server_sock.sendall(bytes([0x80 | 3, 0]))  # opcode 3 is reserved
        with self.assertRaisesRegex(group.GroupError, "unknown websocket opcode"):
            client.recv(10)


def _http_only_reply(sock):
    group._read_http_head(sock)
    group._http_reply(sock, "404 Not Found")


class StateTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.state = make_state(self.dir / "lab")
        self.secret = bytes.fromhex(self.state.data["secret"])
        self.ports = range(52100, 52110)

    def token(self, member="node-a", **overrides):
        fields = dict(group="lab", member=member, tunnel_url=self.state.tunnel_url,
                      expires=time.time() + 3600)
        fields.update(overrides)
        return group.issue_token(self.secret, **fields)

    def test_enrolment_returns_a_key_and_a_port_and_stores_only_a_verifier(self):
        member, key, port = self.state.enrol(self.token(), now=time.time(), port_range=self.ports)
        self.assertEqual((member, port), ("node-a", 52100))
        self.assertEqual(len(key), 64)
        self.assertNotIn(key, (self.dir / "lab" / "group.json").read_text())
        self.assertEqual(self.state.authenticate("node-a", key), 52100)
        self.assertIsNone(self.state.authenticate("node-a", "0" * 64))
        self.assertIsNone(self.state.authenticate("node-b", key))
        self.assertIsNone(self.state.authenticate("node-a", 5))

    def test_a_token_works_once(self):
        token = self.token()
        self.state.enrol(token, now=time.time(), port_range=self.ports)
        with self.assertRaisesRegex(group.GroupError, "already been used"):
            self.state.enrol(token, now=time.time(), port_range=self.ports)

    def test_a_second_member_gets_the_next_port(self):
        self.state.enrol(self.token("node-a"), now=time.time(), port_range=self.ports)
        _, _, port = self.state.enrol(self.token("node-b"), now=time.time(), port_range=self.ports)
        self.assertEqual(port, 52101)

    def test_an_already_enrolled_name_is_refused_with_a_fresh_token(self):
        self.state.enrol(self.token("node-a"), now=time.time(), port_range=self.ports)
        with self.assertRaisesRegex(group.GroupError, "already enrolled"):
            self.state.enrol(self.token("node-a"), now=time.time(), port_range=self.ports)

    def test_token_for_another_tunnel_is_refused(self):
        other = group.issue_token(self.secret, group="lab", member="node-a",
                                  tunnel_url="https://old-tunnel.trycloudflare.com", expires=time.time() + 60)
        with self.assertRaisesRegex(group.GroupError, "different tunnel"):
            self.state.enrol(other, now=time.time(), port_range=self.ports)

    def test_token_for_another_group_is_refused(self):
        other = group.issue_token(self.secret, group="other", member="node-a",
                                  tunnel_url=TUNNEL_URL, expires=time.time() + 60)
        with self.assertRaisesRegex(group.GroupError, "different group"):
            self.state.enrol(other, now=time.time(), port_range=self.ports)

    def test_no_running_relay_means_no_enrolment(self):
        state = make_state(self.dir / "idle", tunnel_url=None)
        secret = bytes.fromhex(state.data["secret"])
        token = group.issue_token(secret, group="lab", member="node-a", tunnel_url=TUNNEL_URL,
                                  expires=time.time() + 60)
        with self.assertRaisesRegex(group.GroupError, "not running"):
            state.enrol(token, now=time.time(), port_range=self.ports)

    def test_expired_token_is_refused(self):
        with self.assertRaisesRegex(group.GroupError, "expired"):
            self.state.enrol(self.token(expires=time.time() - 1), now=time.time(), port_range=self.ports)

    def test_issue_refuses_a_name_already_enrolled(self):
        self.state.enrol(self.token("node-a"), now=time.time(), port_range=self.ports)
        with self.assertRaisesRegex(group.GroupError, "already enrolled"):
            group.issue("lab", "node-a", groups_dir=self.dir)

    def test_issue_needs_a_running_relay(self):
        make_state(self.dir / "idle" / "lab", tunnel_url=None)
        with self.assertRaisesRegex(group.GroupError, "not running"):
            group.issue("lab", "node-x", groups_dir=self.dir / "idle")

    def test_issue_names_the_running_tunnel(self):
        token = group.issue("lab", "node-c", groups_dir=self.dir, now=time.time())
        self.assertEqual(group.read_token(token)[0]["u"], TUNNEL_URL)


class GpuDeviceTests(unittest.TestCase):
    """A member serves a GPU only: the device list is checked before anything is spent or served."""

    def test_a_cpu_only_list_is_refused_with_the_reason(self):
        with self.assertRaisesRegex(group.GroupError, r"no GPU device.*lists CPU.*not fall back to CPU"):
            group.select_gpu_device(Path("/unused"), list_devices=lambda _binary: ["CPU"])

    def test_an_empty_list_is_refused(self):
        with self.assertRaisesRegex(group.GroupError, "lists nothing"):
            group.select_gpu_device(Path("/unused"), list_devices=lambda _binary: [])

    def test_rpc_and_blas_devices_do_not_count_as_a_gpu(self):
        with self.assertRaisesRegex(group.GroupError, "no GPU device"):
            group.select_gpu_device(Path("/unused"), list_devices=lambda _binary: ["RPC0", "BLAS0", "CPU"])

    def test_the_first_gpu_is_chosen_and_cpu_is_ignored(self):
        self.assertEqual(group.select_gpu_device(Path("/unused"), list_devices=lambda _b: ["CPU", "CUDA1", "Vulkan0"]),
                         "CUDA1")

    def test_metal_and_vulkan_names_are_gpus(self):
        self.assertEqual(group.select_gpu_device(Path("/unused"), list_devices=lambda _b: ["MTL0"]), "MTL0")
        self.assertEqual(group.select_gpu_device(Path("/unused"), list_devices=lambda _b: ["Vulkan0"]), "Vulkan0")

    # Captured on this Mac from the installed llama.cpp b11700 (`llama-server --list-devices`).
    # b11752, the current release, prints the same two lines.
    REAL_LLAMA_LISTING = ("Available devices:\n"
                          "  MTL0: Apple M3 (12124 MiB, 12123 MiB free)\n"
                          "  BLAS: Accelerate (0 MiB, 0 MiB free)\n")

    def test_the_real_listing_from_this_mac_parses_to_the_metal_device(self):
        self.assertEqual(group.parse_device_listing(self.REAL_LLAMA_LISTING), ["MTL0", "BLAS"])
        self.assertEqual(group.select_gpu_device(Path("/unused"),
                                                 list_devices=lambda _b: group.parse_device_listing(self.REAL_LLAMA_LISTING)),
                         "MTL0")

    def test_the_real_listing_with_no_gpu_line_is_refused(self):
        cpu_only = "Available devices:\n  CPU: Apple M3 (8 MiB, 8 MiB free)\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n"
        self.assertEqual(group.parse_device_listing(cpu_only), ["CPU", "BLAS"])
        with self.assertRaisesRegex(group.GroupError, "no GPU device.*lists CPU, BLAS.*not fall back to CPU"):
            group.select_gpu_device(Path("/unused"), list_devices=lambda _b: group.parse_device_listing(cpu_only))
        with self.assertRaisesRegex(group.GroupError, "no GPU device"):
            group.select_gpu_device(Path("/unused"), list_devices=lambda _b: group.parse_device_listing("Available devices:\n"))

    def test_the_device_names_come_from_the_lines_and_never_from_a_guess(self):
        # A backend the parser does not know is not a GPU, even if its name sounds like one.
        self.assertEqual(group.parse_device_listing("Available devices:\n  Foo0: Something (1 MiB)\n"), ["Foo0"])
        with self.assertRaisesRegex(group.GroupError, "no GPU device"):
            group.select_gpu_device(Path("/unused"), list_devices=lambda _b: ["Foo0"])

    def test_device_listing_is_read_from_llama_server_beside_the_rpc_binary(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        (root / "llama-server").write_text(
            f"#!{sys.executable}\nimport sys\nassert sys.argv[1:] == ['--list-devices']\n"
            f"sys.stdout.write({self.REAL_LLAMA_LISTING!r})\n")
        (root / "llama-server").chmod(0o755)
        self.assertEqual(group.list_rpc_devices(root / "ggml-rpc-server"), ["MTL0", "BLAS"])

    def test_the_rpc_server_is_never_asked_to_list_devices(self):
        # The real ggml-rpc-server (b11700, b11752) answers --list-devices with "unknown argument"
        # and its usage text, and exits 0. Parsing that text as a device listing finds no GPU.
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        rpc = root / "ggml-rpc-server"
        rpc.write_text(f"#!{sys.executable}\nimport sys\n"
                       "if '--list-devices' in sys.argv:\n"
                       "    print('error: unknown argument: --list-devices', file=sys.stderr)\n"
                       "    print('Usage: ggml-rpc-server [options]', file=sys.stderr)\n"
                       "    print('  -d, --device <dev1,dev2,...>     comma-separated list of devices', file=sys.stderr)\n"
                       "    sys.exit(0)\n")
        rpc.chmod(0o755)
        (root / "llama-server").write_text(f"#!{sys.executable}\nprint({self.REAL_LLAMA_LISTING!r}, end='')\n")
        (root / "llama-server").chmod(0o755)
        self.assertEqual(group.list_rpc_devices(rpc), ["MTL0", "BLAS"])

    def test_a_failed_device_listing_is_refused(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        (root / "llama-server").write_text(f"#!{sys.executable}\nimport sys\nsys.exit(2)\n")
        (root / "llama-server").chmod(0o755)
        with self.assertRaisesRegex(group.GroupError, "could not list its devices"):
            group.list_rpc_devices(root / "ggml-rpc-server")

    def test_a_missing_llama_server_is_refused_without_a_traceback(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        with self.assertRaisesRegex(group.GroupError, "cannot ask llama-server"):
            group.list_rpc_devices(root / "ggml-rpc-server")

    def test_a_cpu_only_member_is_refused_before_it_spends_its_token(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        coordinator = root / "coordinator"
        member_dir = root / "member"
        state = make_state(coordinator / "lab")
        token = group.issue("lab", "node-cpu", groups_dir=coordinator)
        with patch.object(group, "enrol", side_effect=AssertionError("enrolled without a GPU")), \
                patch.object(group, "run_worker", side_effect=AssertionError("served without a GPU")):
            with self.assertRaisesRegex(group.GroupError, "no GPU device"):
                group.join_group(token, rpc_binary=Path("/unused"), groups_dir=member_dir,
                                 list_devices=lambda _b: ["CPU"])
        self.assertFalse(group.member_path(member_dir, "lab").exists())
        # The token is still unspent: the same coordinator can admit it, now that a GPU would be there.
        reloaded = group.GroupState.load(coordinator / "lab" / "group.json")
        self.assertEqual(reloaded.data["used"], {})
        self.assertEqual(state.data["members"], {})


class EndToEndTests(unittest.TestCase):
    """Init, the relay on loopback with a stub tunnel, enrolment over WebSocket, and bytes both ways."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.groups = self.root / "coordinator"
        self.members = self.root / "member"
        group.init_group("lab", groups_dir=self.groups)
        self.state_path = self.groups / "lab" / "group.json"
        self.fake_rpc = self._fake_rpc_binary()
        (self.root / "llama-server").write_text(
            f"#!{sys.executable}\nprint('Available devices:')\nprint('  CUDA0: Fake GPU for tests (1024 MiB, 1024 MiB free)')\n")
        (self.root / "llama-server").chmod(0o755)
        self.relay_logs = []
        self.stop = threading.Event()
        self.addCleanup(self._stop_relay)
        base = free_port()
        self.rpc_ports = range(base, base + 10)
        self.relay_thread = threading.Thread(
            target=group.run_relay, daemon=True,
            kwargs=dict(state_path=self.state_path, log=self.relay_logs.append, stop=self.stop,
                        open_tunnel=stub_tunnel, port_range=self.rpc_ports))
        self.relay_thread.start()
        self.tunnel = self._wait_for_tunnel_url()

    def _stop_relay(self):
        self.stop.set()
        self.relay_thread.join(timeout=10)

    def _fake_rpc_binary(self) -> Path:
        path = self.root / "fake-ggml-rpc-server"
        path.write_text(
            f"#!{sys.executable}\n"
            "import socket, sys, threading\n"
            "if '--list-devices' in sys.argv:\n"
            "    print('error: unknown argument: --list-devices', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "device =sys.argv[sys.argv.index('-d') + 1] if '-d' in sys.argv else ''\n"
            "if not device.startswith('CUDA'):\n"
            "    sys.exit(3)\n"
            "host = sys.argv[sys.argv.index('--host') + 1]\n"
            "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
            "srv = socket.create_server((host, port))\n"
            "def echo(conn):\n"
            "    with conn:\n"
            "        while True:\n"
            "            data = conn.recv(65536)\n"
            "            if not data:\n"
            "                return\n"
            "            conn.sendall(data)\n"
            "while True:\n"
            "    conn, _ = srv.accept()\n"
            "    threading.Thread(target=echo, args=(conn,), daemon=True).start()\n")
        path.chmod(0o755)
        return path

    def _wait_for_tunnel_url(self, timeout: float = 30.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            url = json.loads(self.state_path.read_text()).get("tunnel_url")
            if url:
                return url
            time.sleep(0.05)
        self.fail(f"the relay never published a tunnel URL; log: {self.relay_logs}")

    def test_relay_serves_health_and_refuses_a_plain_request_on_the_link_path(self):
        with urllib.request.urlopen(f"{self.tunnel}{group.HEALTH_PATH}", timeout=5) as resp:
            self.assertEqual(json.loads(resp.read()), {"ok": True})
        with self.assertRaises(urllib.error.HTTPError) as caught:  # the path exists, but a plain GET is not an upgrade
            urllib.request.urlopen(f"{self.tunnel}{group.LINK_PATH}", timeout=5)
        self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{self.tunnel}/elsewhere", timeout=5)
        self.assertEqual(caught.exception.code, 404)

    def test_relay_printed_the_tunnel_url(self):
        self.assertTrue(any(line == f"tunnel URL: {self.tunnel}" for line in self.relay_logs), self.relay_logs)

    def test_member_enrols_over_websocket_and_its_gpu_worker_is_reachable_on_the_coordinator_port(self):
        token = group.issue("lab", "node-a", groups_dir=self.groups)
        stop = threading.Event()
        self.addCleanup(stop.set)
        logs = []
        worker = threading.Thread(
            target=group.join_group, daemon=True,
            kwargs=dict(token=token, rpc_binary=self.fake_rpc, groups_dir=self.members,
                        log=logs.append, stop=stop))
        worker.start()

        port = self._wait_for_member("node-a")
        self.assertIn(port, self.rpc_ports)
        payload = b"rpc-bytes " * 40_000  # forces several frames each way
        self.assertTrue(self._echo_through(port, payload), f"no echo through the link; log: {logs}")
        self.assertTrue(any("GPU device CUDA0" in line for line in logs), logs)

        # The token cannot be spent a second time, and the member keeps its key with owner-only permissions.
        with self.assertRaisesRegex(group.GroupError, "already been used"):
            group.enrol(token, groups_dir=Path(tempfile.mkdtemp()))
        member_file = self.members / "lab" / "member.json"
        self.assertTrue(member_file.exists())
        self.assertEqual(member_file.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(member_file.read_text())["tunnel_url"], self.tunnel)

    def test_expired_token_is_refused_by_the_member_before_connecting(self):
        secret = bytes.fromhex(json.loads(self.state_path.read_text())["secret"])
        stale = group.issue_token(secret, group="lab", member="node-z", tunnel_url=self.tunnel,
                                  expires=time.time() - 5)
        with patch.object(group, "client_connect", side_effect=AssertionError("connected")):
            with self.assertRaisesRegex(group.GroupError, "expired"):
                group.enrol(stale, groups_dir=self.members)

    def _wait_for_member(self, name: str, timeout: float = 30.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            members = json.loads(self.state_path.read_text())["members"]
            if name in members:
                return members[name]["rpc_port"]
            time.sleep(0.1)
        self.fail(f"{name} never enrolled")

    def _echo_through(self, port: int, payload: bytes, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                    sock.sendall(payload)
                    received = bytearray()
                    while len(received) < len(payload):
                        chunk = sock.recv(65536)
                        if not chunk:
                            break
                        received += chunk
                    if bytes(received) == payload:
                        return True
            except OSError:
                pass
            time.sleep(0.2)
        return False


class _StubProcess:
    """Stands in for the cloudflared process: alive until stopped."""

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def stub_tunnel(local_url: str, local_port: int):
    """A tunnel that is the local address itself: reachable from this machine, which is all the test needs."""
    return _StubProcess(), local_url


class RelayTests(unittest.TestCase):
    def test_relay_listens_on_loopback_only(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        make_state(root / "lab")
        with self.assertRaisesRegex(group.GroupError, "127.0.0.1 only"):
            group.Relay(root / "lab" / "group.json", "0.0.0.0", 0)

    def test_run_relay_hands_the_tunnel_the_local_relay_and_its_host_header(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        group.init_group("lab", groups_dir=root)
        seen = {}

        def tunnel(local_url, local_port):
            seen["target"] = local_url
            seen["port"] = local_port
            return _StubProcess(), local_url

        stop = threading.Event()
        thread = threading.Thread(target=group.run_relay, daemon=True,
                                  kwargs=dict(state_path=root / "lab" / "group.json", stop=stop,
                                              log=lambda _t: None, open_tunnel=tunnel))
        thread.start()
        deadline = time.monotonic() + 30
        while "target" not in seen and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(seen["target"], f"http://127.0.0.1:{seen['port']}")
        stop.set()
        thread.join(timeout=10)
        # the tunnel URL is cleared on the way out, so issue refuses until the relay runs again
        self.assertIsNone(json.loads((root / "lab" / "group.json").read_text())["tunnel_url"])


class _FakeClock:
    """Stands in for the time module inside group.py: monotonic() moves only when a test advances it."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _HealthTicks(threading.Event):
    """Stands in for the relay's stop event: each wait() is one health-check interval on the fake clock."""

    def __init__(self, clock, ticks):
        super().__init__()
        self.clock = clock
        self.left = ticks

    def wait(self, timeout=None):
        if self.left == 0:
            return True
        self.left -= 1
        self.clock.now += timeout
        return False


class TunnelGraceTests(unittest.TestCase):
    """A quick tunnel is replaced only after it has been unreachable for the whole grace period.

    The health check is the real tunnel_is_healthy, fed the errors urllib raises: a DNS
    failure for a fresh hostname is a URLError wrapping gaierror, the same as a refused
    connection to a dead tunnel. Each outcome is one health check, one interval apart.
    """

    DNS_DOWN = urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))
    REFUSED = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
    HEALTHY = None

    def run_relay_checks(self, outcomes):
        """Run the relay for one health check per outcome. Returns (tunnels opened, relay log)."""
        import join
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        group.init_group("lab", groups_dir=root)
        opened = []

        def tunnel(local_url, local_port):
            opened.append(local_url)
            return _StubProcess(), f"https://tunnel-{len(opened)}.trycloudflare.com"

        script = list(outcomes)

        def http_json(method, url, timeout=None, headers=None):
            outcome = script.pop(0)
            if outcome is None:
                return {"ok": True}
            raise outcome

        clock = _FakeClock()
        logs = []
        with patch.object(group, "time", clock), \
                patch.object(join, "http_json", side_effect=http_json), \
                patch.object(join, "wait_for_tunnel", return_value=False):
            group.run_relay(root / "lab" / "group.json", stop=_HealthTicks(clock, len(outcomes)),
                            log=logs.append, open_tunnel=tunnel)
        self.assertEqual(script, [], "the relay did not run one health check per outcome")
        return len(opened), logs

    def test_dns_failures_inside_the_grace_period_do_not_rotate_the_tunnel(self):
        opened, logs = self.run_relay_checks([self.DNS_DOWN] * 4)  # 480s of DNS failure, under the 600s grace
        self.assertEqual(opened, 1)
        self.assertFalse(any("restarting" in line for line in logs), logs)

    def test_dns_recovery_restarts_the_grace_period(self):
        opened, _ = self.run_relay_checks([self.DNS_DOWN] * 4 + [self.HEALTHY] + [self.DNS_DOWN] * 4)
        self.assertEqual(opened, 1)  # 4 failures, a success, then 4 more: never 600s in a row

    def test_an_unreachable_tunnel_is_not_rotated_before_the_grace_period_ends(self):
        opened, _ = self.run_relay_checks([self.REFUSED] * 4)  # 480s
        self.assertEqual(opened, 1)

    def test_a_tunnel_unreachable_for_the_whole_grace_period_is_rotated(self):
        opened, logs = self.run_relay_checks([self.REFUSED] * 5)  # 600s
        self.assertEqual(opened, 2)
        self.assertIn("tunnel URL: https://tunnel-2.trycloudflare.com", logs)

    def test_dns_failure_for_the_whole_grace_period_is_rotated_as_a_dead_tunnel(self):
        # Past the grace period a failure cannot be told from a dead tunnel, so it is replaced.
        opened, _ = self.run_relay_checks([self.DNS_DOWN] * 5)
        self.assertEqual(opened, 2)


class CommandLineTests(unittest.TestCase):
    def test_members_on_an_unknown_group_is_a_clean_error(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        with patch.object(group, "GROUPS_DIR", root), patch.object(sys, "stderr", io.StringIO()) as err:
            self.assertEqual(group.main(["members", "--group", "nothere"]), 1)
        self.assertIn("no group at", err.getvalue())

    def test_init_takes_no_relay_address_any_more(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        with patch.object(group, "GROUPS_DIR", root), patch.object(sys, "stdout", io.StringIO()):
            self.assertEqual(group.main(["init", "--group", "lab"]), 0)
            self.assertIsNone(json.loads((root / "lab" / "group.json").read_text())["tunnel_url"])
        with patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                group.main(["init", "--group", "lab", "--relay", "192.0.2.10:7443"])

    def test_relay_command_starts_the_relay_for_the_named_group(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        group.init_group("lab", groups_dir=root)
        with patch.object(group, "GROUPS_DIR", root), patch.object(group, "run_relay") as run, \
                patch.object(sys, "stdout", io.StringIO()):
            self.assertEqual(group.main(["relay", "--group", "lab", "--port", "0"]), 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], root / "lab" / "group.json")
        self.assertEqual(run.call_args.kwargs["port"], 0)

    def test_issue_on_an_unknown_group_is_a_clean_error(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        with patch.object(group, "GROUPS_DIR", root), patch.object(sys, "stderr", io.StringIO()) as err:
            self.assertEqual(group.main(["issue", "--group", "nothere", "--member", "node-a"]), 1)
        self.assertIn("no group at", err.getvalue())


class JoinEntryPointTests(unittest.TestCase):
    """`join --group-token` routes to the member flow, and refuses combinations that make no sense."""

    def setUp(self):
        import join as join_module
        self.join = join_module

    def run_join(self, argv):
        # join.main() reconfigures stdout, so give it a real text stream.
        with patch.object(sys, "argv", ["join.py", "--no-update", *argv]), \
                open(os.devnull, "w") as out, patch.object(sys, "stdout", out):
            return self.join.main()

    def test_group_token_runs_the_member_flow_and_nothing_else(self):
        with patch.object(self.join, "run_group_member") as member, \
                patch.object(self.join, "resolve_model", side_effect=AssertionError("gateway path")):
            self.run_join(["--group-token", "cng1.x.y"])
        member.assert_called_once_with("cng1.x.y")

    def test_group_token_cannot_be_combined_with_gateway_or_local_modes(self):
        for extra in (["--model", "qwen"], ["--local-only", "--distributed-config", "c.json"],
                      ["--lan"], ["--permanent"], ["--list-catalogue"], ["--port", "11437"]):
            with self.subTest(extra=extra):
                with patch.object(self.join, "run_group_member") as member, \
                        patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                    self.run_join(["--group-token", "cng1.x.y", *extra])
                member.assert_not_called()

    def test_run_group_member_reports_a_refused_token_without_a_traceback(self):
        fake_runtime = MagicMock()
        fake_runtime.FetchFailed = RuntimeError
        with patch.object(self.join, "_require_runtime", return_value=fake_runtime), \
                patch.object(group, "join_group", side_effect=group.GroupError("expired token")), \
                patch.object(self.join, "die", side_effect=SystemExit(1)) as die:
            with self.assertRaises(SystemExit):
                self.join.run_group_member("cng1.x.y")
        die.assert_called_once_with("expired token")


if __name__ == "__main__":
    unittest.main()
