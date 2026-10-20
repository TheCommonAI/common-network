"""Shared-secret tests: the token sources, the 401 on a missing or wrong token, and refusing to start without one.

The token values here are fixed test strings. They are never printed by the code under test.
"""
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from pedal.net import auth, protocol, server
from pedal.net.client import BlockClient, RemoteError
from pedal.net.server import BlockServer, StubBlock
from pedal.tests.test_net_server import hidden_tensor

TOKEN = "test-token-0123456789abcdef"
WRONG = "wrong-token-0123456789abcdef"


def without_env_token():
    env = dict(os.environ)
    env.pop(auth.TOKEN_ENV, None)
    return patch.dict(os.environ, env, clear=True)


def request_status(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.read()


class TokenOnTheServer(unittest.TestCase):
    def setUp(self):
        self.block = StubBlock(0, 4)
        self.server = BlockServer(self.block, "127.0.0.1", 0, token=TOKEN).start()
        self.addCleanup(self.server.stop)
        self.url = self.server.url

    def test_missing_token_is_401_on_health_and_on_post(self):
        self.assertEqual(request_status(self.url + "/health")[0], 401)
        status, body = request_status(self.url + "/session", "POST", {"id": "s1"})
        self.assertEqual(status, 401)
        self.assertIn(b"bearer", body)
        self.assertEqual(self.block.calls, [], "a refused request must not reach the block")

    def test_wrong_token_is_401(self):
        headers = {"Authorization": f"Bearer {WRONG}"}
        self.assertEqual(request_status(self.url + "/health", headers=headers)[0], 401)
        status, _ = request_status(self.url + "/session", "POST", {"id": "s1"}, headers)
        self.assertEqual(status, 401)
        self.assertEqual(self.block.calls, [])

    def test_wrong_scheme_or_partial_token_is_401(self):
        for header in (TOKEN, f"Basic {TOKEN}", f"Bearer {TOKEN[:-1]}", f"Bearer {TOKEN}x"):
            with self.subTest(header=header[:12]):
                status, _ = request_status(self.url + "/health", headers={"Authorization": header})
                self.assertEqual(status, 401)

    def test_correct_token_is_accepted(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        status, _ = request_status(self.url + "/health", headers=headers)
        self.assertEqual(status, 200)
        status, _ = request_status(self.url + "/session", "POST", {"id": "s1"}, headers)
        self.assertEqual(status, 200)

    def test_the_token_never_appears_in_a_reply(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        for status, body in (request_status(self.url + "/health", headers=headers),
                             request_status(self.url + "/health", headers={"Authorization": f"Bearer {WRONG}"})):
            self.assertNotIn(TOKEN.encode(), body)
            self.assertNotIn(WRONG.encode(), body)

    def test_a_short_or_spaced_token_is_refused_at_construction(self):
        for bad in ("short", "has space in the middle of it", ""):
            with self.subTest(token=bad[:8]):
                with self.assertRaises(auth.TokenError):
                    BlockServer(StubBlock(0, 1), "127.0.0.1", 0, token=bad)


class ClientTokens(unittest.TestCase):
    def setUp(self):
        self.block = StubBlock(0, 4)
        self.server = BlockServer(self.block, "127.0.0.1", 0, token=TOKEN).start()
        self.addCleanup(self.server.stop)
        self.url = self.server.url

    def test_client_without_a_token_gets_401(self):
        client = BlockClient(token=None, retries=3, sleep=lambda s: None)
        with self.assertRaises(RemoteError) as caught:
            client.open(self.url, "s1")
        self.assertEqual(caught.exception.status, 401)
        self.assertEqual(self.block.calls, [])

    def test_client_with_a_wrong_token_gets_401_and_is_not_retried(self):
        sleeps = []
        client = BlockClient(token=WRONG, retries=3, sleep=sleeps.append)
        with self.assertRaises(RemoteError) as caught:
            client.forward(self.url, "s1", hidden_tensor(), 0)
        self.assertEqual(caught.exception.status, 401)
        self.assertEqual(sleeps, [], "a 401 is an answer, not a transient failure")

    def test_client_with_the_right_token_works(self):
        client = BlockClient(token=TOKEN)
        client.open(self.url, "s1")
        out = client.forward(self.url, "s1", hidden_tensor(seed=4), 0)
        self.assertEqual(out, hidden_tensor(seed=4))
        client.close(self.url, "s1")

    def test_client_reads_the_same_source_as_the_server(self):
        with patch.dict(os.environ, {auth.TOKEN_ENV: TOKEN}):
            client = BlockClient()
            self.assertEqual(client.token, TOKEN)
            client.open(self.url, "s1")
        with without_env_token():
            self.assertIsNone(BlockClient().token)

    def test_a_wrong_token_in_the_error_message_is_not_echoed(self):
        client = BlockClient(token=WRONG, retries=0)
        with self.assertRaises(RemoteError) as caught:
            client.open(self.url, "s1")
        self.assertNotIn(WRONG, str(caught.exception))
        self.assertNotIn(TOKEN, str(caught.exception))


class TokenSources(unittest.TestCase):
    def test_env_var_is_read(self):
        self.assertEqual(auth.load_token(environ={auth.TOKEN_ENV: TOKEN}), TOKEN)

    def test_file_is_read_and_trimmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            path.write_text(f"  {TOKEN}\n", encoding="utf-8")
            self.assertEqual(auth.load_token(path, environ={}), TOKEN)

    def test_no_source_means_no_token(self):
        self.assertIsNone(auth.load_token(environ={}))
        self.assertIsNone(auth.load_token(environ={auth.TOKEN_ENV: "   "}))

    def test_env_and_file_together_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            path.write_text(TOKEN, encoding="utf-8")
            with self.assertRaises(auth.TokenError) as caught:
                auth.load_token(path, environ={auth.TOKEN_ENV: WRONG})
        self.assertNotIn(WRONG, str(caught.exception))

    def test_short_token_is_refused_without_echoing_it(self):
        with self.assertRaises(auth.TokenError) as caught:
            auth.load_token(environ={auth.TOKEN_ENV: "short"})
        self.assertNotIn("short", str(caught.exception))

    def test_unreadable_file_is_a_token_error(self):
        with self.assertRaises(auth.TokenError):
            auth.load_token(Path(tempfile.gettempdir()) / "pedal-no-such-token-file", environ={})


class RefusesWithoutAToken(unittest.TestCase):
    def test_the_server_command_refuses_a_real_block_without_a_token(self):
        with without_env_token():
            code = server.main(["--block", "nowhere:factory", "--layers", "0", "4"])
        self.assertEqual(code, 2)

    def test_the_server_command_refuses_a_bad_token_source(self):
        with patch.dict(os.environ, {auth.TOKEN_ENV: "short"}):
            self.assertEqual(server.main(["--block", "nowhere:factory", "--layers", "0", "4"]), 2)

    def test_the_launcher_refuses_a_real_block_without_a_token(self):
        from pedal.net import tunnel
        with without_env_token():
            code = tunnel.main(["--block", "nowhere:factory", "--layers", "0", "4", "--out", os.devnull])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
