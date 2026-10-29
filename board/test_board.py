#!/usr/bin/env python3
"""Tests for the message board: storage rules first, then the HTTP surface."""
import json
import threading
import unittest
import urllib.error
import urllib.request

from board import Board, BODY_MAX, NAME_MAX, PAYLOAD_MAX, make_server


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_post_and_list_newest_first(self):
        first = self.board.post("ana", "meet at the gate")
        second = self.board.post("ben", "running late")
        messages = self.board.latest()
        self.assertEqual([m["id"] for m in messages], [second["id"], first["id"]])
        self.assertEqual(messages[0]["body"], "running late")

    def test_blank_name_becomes_anon(self):
        self.assertEqual(self.board.post("   ", "hi")["name"], "anon")
        self.assertEqual(self.board.post(None, "hi")["name"], "anon")

    def test_body_is_stripped_and_required(self):
        self.assertEqual(self.board.post("a", "  hello  ")["body"], "hello")
        with self.assertRaises(ValueError):
            self.board.post("a", "   ")
        with self.assertRaises(ValueError):
            self.board.post("a", None)

    def test_length_limits(self):
        self.board.post("a" * NAME_MAX, "b" * BODY_MAX)
        with self.assertRaises(ValueError):
            self.board.post("a" * (NAME_MAX + 1), "ok")
        with self.assertRaises(ValueError):
            self.board.post("a", "b" * (BODY_MAX + 1))

    def test_non_text_and_nul_rejected(self):
        with self.assertRaises(ValueError):
            self.board.post(["a"], "ok")
        with self.assertRaises(ValueError):
            self.board.post("a", "bad\x00byte")

    def test_latest_respects_limit(self):
        for i in range(5):
            self.board.post("a", f"m{i}")
        self.assertEqual(len(self.board.latest(2)), 2)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = Board()
        cls.server = make_server(cls.board, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as res:
                return res.status, res.headers, res.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers, err.read()

    def test_page_and_assets_served_with_security_headers(self):
        for path, ctype in (("/", "text/html"), ("/app.js", "javascript"), ("/app.css", "text/css")):
            status, headers, body = self.request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertIn(ctype, headers["Content-Type"])
            self.assertTrue(body)
            self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_page_uses_brand_wordmark(self):
        _, _, body = self.request("GET", "/")
        self.assertIn(b"common<span>.</span>", body)

    def test_post_then_list(self):
        status, _, body = self.request("POST", "/api/messages", {"name": "ana", "body": "hi there"})
        self.assertEqual(status, 201)
        posted = json.loads(body)["message"]
        self.assertEqual(posted["name"], "ana")

        status, _, body = self.request("GET", "/api/messages?limit=5")
        self.assertEqual(status, 200)
        messages = json.loads(body)["messages"]
        self.assertEqual(messages[0]["id"], posted["id"])
        self.assertEqual(messages[0]["body"], "hi there")

    def test_list_limit_is_clamped_and_validated(self):
        status, _, _ = self.request("GET", "/api/messages?limit=abc")
        self.assertEqual(status, 400)
        status, _, body = self.request("GET", "/api/messages?limit=0")
        self.assertEqual(status, 200)
        self.assertLessEqual(len(json.loads(body)["messages"]), 1)

    def test_post_rejects_bad_input(self):
        status, _, body = self.request("POST", "/api/messages", {"name": "a", "body": "  "})
        self.assertEqual(status, 400)
        self.assertIn("write something first", json.loads(body)["error"])

        status, _, _ = self.request("POST", "/api/messages", raw=b"not json")
        self.assertEqual(status, 400)

        status, _, _ = self.request("POST", "/api/messages", raw=b"[1, 2]")
        self.assertEqual(status, 400)

    def test_post_rejects_oversized_payload(self):
        status, _, _ = self.request("POST", "/api/messages", raw=b"x" * (PAYLOAD_MAX + 1))
        self.assertEqual(status, 413)

    def test_unknown_routes_404(self):
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/nope", {"body": "x"})[0], 404)

    def test_health(self):
        status, _, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})


if __name__ == "__main__":
    unittest.main()
