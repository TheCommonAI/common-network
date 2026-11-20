#!/usr/bin/env python3
"""Tests for the portable Python board: auth, /admin-mes paths, storage."""
import base64
import http.client
import json
import threading
import unittest
import urllib.error
import urllib.request

from portable import Board, is_authorized, make_server


class AuthTests(unittest.TestCase):
    def test_accepts_password_with_any_username(self):
        token = base64.b64encode(b"common:s3cret").decode()
        self.assertTrue(is_authorized(f"Basic {token}", "s3cret"))
        token = base64.b64encode(b":s3cret").decode()
        self.assertTrue(is_authorized(f"Basic {token}", "s3cret"))

    def test_rejects_wrong_or_missing(self):
        token = base64.b64encode(b"common:wrong").decode()
        self.assertFalse(is_authorized(f"Basic {token}", "s3cret"))
        self.assertFalse(is_authorized(None, "s3cret"))
        self.assertFalse(is_authorized("", "s3cret"))
        self.assertFalse(is_authorized("Bearer s3cret", "s3cret"))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_newest_first_and_anon(self):
        first = self.board.post("ana", "meet at the gate")
        second = self.board.post("  ", "running late")
        messages = self.board.latest()
        self.assertEqual([m["id"] for m in messages], [second["id"], first["id"]])
        self.assertEqual(second["name"], "anon")


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = Board()
        cls.password = "s3cret"
        cls.server = make_server(cls.board, cls.password, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, password="s3cret", raw=None, headers=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if password is not None:
            token = base64.b64encode(f"x:{password}".encode()).decode()
            req.add_header("Authorization", f"Basic {token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)
        try:
            with urllib.request.urlopen(req) as res:
                return res.status, res.headers, res.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers, err.read()

    def test_gate_blocks_without_password(self):
        status, headers, _ = self.request("GET", "/admin-mes/", password=None)
        self.assertEqual(status, 401)
        self.assertIn("Basic", headers["WWW-Authenticate"])

    def test_page_and_api_under_prefix(self):
        status, _, body = self.request("GET", "/admin-mes/")
        self.assertEqual(status, 200)
        self.assertIn(b"common<span>.</span>", body)

        status, _, _ = self.request("POST", "/admin-mes/api/messages", {"name": "ana", "body": "hi"})
        self.assertEqual(status, 201)
        status, _, body = self.request("GET", "/admin-mes/api/messages?limit=1")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["messages"][0]["body"], "hi")

    def test_rejects_negative_content_length_without_reading(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        token = base64.b64encode(b"x:s3cret").decode()
        conn.putrequest("POST", "/api/messages")
        conn.putheader("Authorization", f"Basic {token}")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", "-1")
        conn.endheaders()
        conn.send(b'{"body":"x"}')
        res = conn.getresponse()
        status = res.status
        res.read()
        conn.close()
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
