import os
import stat
import tempfile
import unittest

from commonnet.mesh import records
from commonnet.mesh.records import Identity, RecordError, is_live, verify_record

FIELDS = {"network": "n1", "url": "http://127.0.0.1:9", "token": "tok", "window": [0, 9], "backend": "mlx",
          "state": "up", "gib": 10.5}


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.who = Identity.generate()

    def test_signed_record_verifies(self):
        r = self.who.sign_record(FIELDS, now=1000.0)
        self.assertTrue(verify_record(r))
        self.assertEqual(r["peer"], records.peer_id(self.who.pub))
        self.assertEqual(len(r["peer"]), 32)

    def test_tampering_with_any_field_fails(self):
        r = self.who.sign_record(FIELDS, now=1000.0)
        for key, value in (("window", [0, 36]), ("url", "http://evil.example"), ("token", "x"), ("state", "leaving"),
                           ("seq", r["seq"] + 1), ("ts", r["ts"] + 1), ("gib", 99.0), ("network", "n2")):
            with self.subTest(key=key):
                self.assertFalse(verify_record({**r, key: value}))

    def test_wrong_peer_id_or_missing_signature_fails(self):
        r = self.who.sign_record(FIELDS, now=1000.0)
        other = Identity.generate()
        self.assertFalse(verify_record({**r, "peer": other.peer}))
        self.assertFalse(verify_record({k: v for k, v in r.items() if k != "sig"}))
        self.assertFalse(verify_record({**r, "pub": other.pub_b64}))  # signed by someone else

    def test_garbage_is_refused_not_raised(self):
        for bad in (None, 5, "x", {}, {"v": 1}):
            self.assertFalse(verify_record(bad))
        r = self.who.sign_record(FIELDS, now=1000.0)
        self.assertFalse(verify_record({**r, "window": "no"}))
        self.assertFalse(verify_record({**r, "url": "ftp://x"}))
        self.assertFalse(verify_record({**r, "state": "zombie"}))

    def test_seq_is_strictly_increasing_for_one_identity(self):
        a = self.who.sign_record(FIELDS, now=1000.0)
        b = self.who.sign_record(FIELDS, now=1000.0)
        self.assertGreater(b["seq"], a["seq"])

    def test_liveness(self):
        r = self.who.sign_record(FIELDS, now=1000.0)
        self.assertTrue(is_live(r, now=1030.0))
        self.assertTrue(is_live(r, now=1060.0))
        self.assertFalse(is_live(r, now=1061.0))  # stale
        self.assertFalse(is_live(r, now=1000.0 - 121.0))  # dated too far ahead
        self.assertTrue(is_live(r, now=1000.0 - 100.0))  # within the skew allowance
        self.assertFalse(is_live(r, now=1030.0, network="n2"))
        leaving = self.who.sign_record({**FIELDS, "state": "leaving"}, now=1000.0)
        loading = self.who.sign_record({**FIELDS, "state": "loading", "window": None}, now=1000.0)
        self.assertFalse(is_live(leaving, now=1001.0))
        self.assertFalse(is_live(loading, now=1001.0))

    def test_identity_persists_and_key_file_is_private(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "net", "identity.key")
            first = Identity.load_or_create(path)
            second = Identity.load_or_create(path)
            self.assertEqual(first.peer, second.peer)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertTrue(verify_record(second.sign_record(FIELDS)))

    def test_unreadable_key_file_is_an_error_not_a_new_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "identity.key")
            open(path, "w").write("not a key")
            with self.assertRaises(RecordError):
                Identity.load_or_create(path)


if __name__ == "__main__":
    unittest.main()
