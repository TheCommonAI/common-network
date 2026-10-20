"""Offline tests for distributed group enrolment, the outbound worker framing, and the link.

The end-to-end test runs a real TLS relay and member agent on loopback, with a
small Python stand-in for ggml-rpc-server. It touches no RPC port other than the
ephemeral ones it picks, and never starts llama.cpp or a model.
"""
import io
import json
import os
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import group

OPENSSL = shutil.which("openssl")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def make_state(directory: Path, relay: str = "127.0.0.1:7443") -> group.GroupState:
    """A coordinator record without openssl: enrolment does not need a certificate."""
    data = {"v": 1, "group": "lab", "relay": relay, "cert_fp": "ab" * 32,
            "secret": secrets.token_hex(32), "members": {}, "used": {}}
    state = group.GroupState(directory / "group.json", data)
    state.save()
    return state


class TokenTests(unittest.TestCase):
    SECRET = bytes(range(32))
    NOW = 1_800_000_000.0

    def token(self, **overrides):
        fields = dict(group="lab", member="node-a", relay="192.0.2.10:7443", cert_fp="cd" * 32,
                      expires=self.NOW + 3600)
        fields.update(overrides)
        return group.issue_token(self.SECRET, **fields)

    def test_valid_token_round_trips_its_fields(self):
        payload = group.check_token(self.SECRET, self.token(), self.NOW)
        self.assertEqual((payload["g"], payload["m"], payload["r"], payload["f"]),
                         ("lab", "node-a", "192.0.2.10:7443", "cd" * 32))

    def test_each_token_gets_its_own_nonce(self):
        first = group.read_token(self.token())[0]["n"]
        second = group.read_token(self.token())[0]["n"]
        self.assertNotEqual(first, second)

    def test_wrong_secret_is_rejected(self):
        with self.assertRaisesRegex(group.GroupError, "not valid for this group"):
            group.check_token(bytes(32), self.token(), self.NOW)

    def test_tampered_payload_is_rejected(self):
        token = self.token()
        prefix, body, signature = token.split(".")
        forged = group._b64e(json.dumps({**json.loads(group._b64d(body)), "m": "intruder"},
                                        sort_keys=True, separators=(",", ":")).encode())
        with self.assertRaisesRegex(group.GroupError, "not valid for this group"):
            group.check_token(self.SECRET, f"{prefix}.{forged}.{signature}", self.NOW)

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

    def test_wrong_version_and_bad_fields_are_refused(self):
        body = json.dumps({"v": 2, "g": "lab", "m": "node-a", "r": "h:1", "f": "cd" * 32, "x": 1, "n": "n"}).encode()
        with self.assertRaisesRegex(group.GroupError, "version"):
            group.read_token("cng1." + group._b64e(body) + "." + group._b64e(b"s"))
        body = json.dumps({"v": 1, "g": "lab", "m": "node-a", "r": "no-port", "f": "cd" * 32, "x": 1, "n": "n"}).encode()
        with self.assertRaisesRegex(group.GroupError, "host:port"):
            group.read_token("cng1." + group._b64e(body) + "." + group._b64e(b"s"))

    def test_names_are_validated_when_issuing(self):
        with self.assertRaises(group.GroupError):
            self.token(member="Not Valid!")
        with self.assertRaises(group.GroupError):
            self.token(group="")


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


class StateTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.state = make_state(self.dir / "lab")
        self.secret = bytes.fromhex(self.state.data["secret"])
        self.ports = range(52100, 52110)

    def token(self, member="node-a", **overrides):
        fields = dict(group="lab", member=member, relay=self.state.relay, cert_fp=self.state.cert_fp,
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

    def test_token_for_another_group_or_coordinator_is_refused(self):
        other = group.issue_token(self.secret, group="lab", member="node-a", relay="203.0.113.9:7443",
                                  cert_fp=self.state.cert_fp, expires=time.time() + 60)
        with self.assertRaisesRegex(group.GroupError, "different group or coordinator"):
            self.state.enrol(other, now=time.time(), port_range=self.ports)

    def test_expired_token_is_refused(self):
        with self.assertRaisesRegex(group.GroupError, "expired"):
            self.state.enrol(self.token(expires=time.time() - 1), now=time.time(), port_range=self.ports)

    def test_issue_refuses_a_name_already_enrolled(self):
        self.state.enrol(self.token("node-a"), now=time.time(), port_range=self.ports)
        with self.assertRaisesRegex(group.GroupError, "already enrolled"):
            group.issue("lab", "node-a", groups_dir=self.dir)


@unittest.skipUnless(OPENSSL, "openssl is needed to make the coordinator certificate")
class EndToEndTests(unittest.TestCase):
    """Init, issue, enrol over pinned TLS, the agent connects, and bytes cross to the member and back."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.groups = self.root / "coordinator"
        self.members = self.root / "member"
        relay_port = free_port()
        self.relay_addr = f"127.0.0.1:{relay_port}"
        group.init_group("lab", self.relay_addr, groups_dir=self.groups)
        base = free_port()
        self.rpc_ports = range(base, base + 10)
        self.relay = group.Relay(self.groups / "lab" / "group.json", "127.0.0.1", relay_port,
                                 port_range=self.rpc_ports, log=lambda _text: None)
        self.relay.start()
        self.addCleanup(self.relay.close)
        self.fake_rpc = self._fake_rpc_binary()

    def _fake_rpc_binary(self) -> Path:
        path = self.root / "fake-ggml-rpc-server"
        path.write_text(
            f"#!{sys.executable}\n"
            "import socket, sys, threading\n"
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

    def test_member_enrols_and_its_worker_is_reachable_on_the_coordinator_port(self):
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

        # The member never reveals its key or its RPC port beyond the local machine,
        # and the token cannot be spent a second time.
        with self.assertRaisesRegex(group.GroupError, "already been used"):
            group.enrol(token, groups_dir=Path(tempfile.mkdtemp()))
        self.assertTrue((self.members / "lab" / "member.json").exists())
        mode = (self.members / "lab" / "member.json").stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_pin_mismatch_refuses_enrolment_before_any_token_is_spent(self):
        secret = bytes.fromhex(json.loads((self.groups / "lab" / "group.json").read_text())["secret"])
        bad = group.issue_token(secret, group="lab", member="node-x", relay=self.relay_addr,
                                cert_fp="00" * 32, expires=time.time() + 60)
        with self.assertRaisesRegex(group.GroupError, "does not match the group token"):
            group.enrol(bad, groups_dir=self.members)
        state = group.GroupState.load(self.groups / "lab" / "group.json")
        self.assertEqual(state.data["used"], {})

    def test_expired_token_is_refused_by_the_member_before_connecting(self):
        secret = bytes.fromhex(json.loads((self.groups / "lab" / "group.json").read_text())["secret"])
        stale = group.issue_token(secret, group="lab", member="node-y", relay=self.relay_addr,
                                  cert_fp=group.GroupState.load(self.groups / "lab" / "group.json").cert_fp,
                                  expires=time.time() - 5)
        with patch.object(group, "client_connect", side_effect=AssertionError("connected")):
            with self.assertRaisesRegex(group.GroupError, "expired"):
                group.enrol(stale, groups_dir=self.members)

    def _wait_for_member(self, name: str, timeout: float = 30.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            members = group.GroupState.load(self.groups / "lab" / "group.json").data["members"]
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


class CommandLineTests(unittest.TestCase):
    def test_members_on_an_unknown_group_is_a_clean_error(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        with patch.object(group, "GROUPS_DIR", root), patch.object(sys, "stderr", io.StringIO()) as err:
            self.assertEqual(group.main(["members", "--group", "nothere"]), 1)
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
