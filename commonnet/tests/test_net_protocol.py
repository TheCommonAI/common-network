"""Protocol tests: tensor and envelope round trips, and the strictness of shape, dtype and size."""
import base64
import json
import unittest

from commonnet.net import protocol
from commonnet.net.protocol import ProtocolError, Tensor


def tensor(shape=(1, 2, 8), dtype="float16") -> Tensor:
    count = protocol.numel(shape) * protocol.DTYPE_WIDTH[dtype]
    data = bytes(range(256)) * (count // 256 + 1)
    return Tensor(shape, dtype, data[:count])


class TensorRoundTrip(unittest.TestCase):
    def test_float16_and_float32_round_trip(self):
        for shape, dtype in (((1, 1, 8), "float16"), ((2, 3, 16), "float16"), ((1, 4, 4), "float32")):
            with self.subTest(shape=shape, dtype=dtype):
                original = tensor(shape, dtype)
                wire = json.loads(json.dumps(protocol.encode_tensor(original)))
                self.assertEqual(protocol.decode_tensor(wire), original)

    def test_wire_fields_and_base64_of_raw_bytes(self):
        original = tensor((1, 1, 4), "float32")
        wire = protocol.encode_tensor(original)
        self.assertEqual(set(wire), {"shape", "dtype", "data"})
        self.assertEqual(wire["shape"], [1, 1, 4])
        self.assertEqual(base64.b64decode(wire["data"]), original.data)

    def test_4096_hidden_token_is_8kb_in_float16(self):
        token = Tensor((1, 1, 4096), "float16", bytes(4096 * 2))
        self.assertEqual(len(token.data), 8192)
        self.assertEqual(len(base64.b64decode(protocol.encode_tensor(token)["data"])), 8192)

    def test_tensor_constructor_rejects_bad_data_length(self):
        with self.assertRaises(ProtocolError):
            Tensor((1, 1, 8), "float16", bytes(15))
        with self.assertRaises(ProtocolError):
            Tensor((1, 1, 8), "float16", bytes(17))


class TensorStrictness(unittest.TestCase):
    def good(self, **overrides):
        obj = protocol.encode_tensor(tensor((1, 1, 8)))
        obj.update(overrides)
        return obj

    def assertRejected(self, obj):
        with self.assertRaises(ProtocolError):
            protocol.decode_tensor(obj)

    def test_two_dimensional_or_four_dimensional_shape_is_rejected(self):
        self.assertRejected(self.good(shape=[1, 8]))
        self.assertRejected(self.good(shape=[1, 1, 1, 8]))

    def test_shape_entries_must_be_positive_ints(self):
        for bad in ([1, 1, 0], [1, 1, -8], [1, 1, 8.0], [1, 1, "8"], [1, 1, True], [1, 1, None]):
            with self.subTest(shape=bad):
                self.assertRejected(self.good(shape=bad))

    def test_shape_must_be_a_list(self):
        self.assertRejected(self.good(shape="1x1x8"))
        self.assertRejected(self.good(shape={"0": 1, "1": 1, "2": 8}))

    def test_dtype_must_be_float16_or_float32(self):
        for bad in ("bfloat16", "float64", "int8", "FLOAT16", "", 16, None):
            with self.subTest(dtype=bad):
                self.assertRejected(self.good(dtype=bad))

    def test_byte_count_must_match_shape_and_dtype(self):
        short = base64.b64encode(bytes(8)).decode()
        self.assertRejected(self.good(data=short))  # 8 bytes, shape needs 16 for float16
        self.assertRejected(self.good(dtype="float32"))  # same 16 bytes, but float32 needs 32

    def test_data_must_be_valid_base64(self):
        self.assertRejected(self.good(data="not base64!!"))
        self.assertRejected(self.good(data=12345))
        self.assertRejected(self.good(data=base64.b64encode(bytes(16)).decode() + "\n"))

    def test_missing_and_unknown_fields_are_rejected(self):
        obj = self.good()
        missing = dict(obj)
        del missing["dtype"]
        self.assertRejected(missing)
        self.assertRejected(dict(obj, extra=1))

    def test_non_object_is_rejected(self):
        self.assertRejected([1, 2, 3])
        self.assertRejected("tensor")

    def test_encode_refuses_non_tensors(self):
        with self.assertRaises(TypeError):
            protocol.encode_tensor({"shape": [1, 1, 8]})


class EnvelopeRoundTrip(unittest.TestCase):
    def test_forward_request_round_trip(self):
        hidden = tensor((1, 3, 16))
        body = json.loads(json.dumps(protocol.encode_forward_request("sess-1", hidden, 7)))
        self.assertEqual(set(body), {"id", "start_pos", "shape", "dtype", "data"})
        session_id, start_pos, decoded = protocol.decode_forward_request(body)
        self.assertEqual((session_id, start_pos, decoded), ("sess-1", 7, hidden))

    def test_forward_reply_round_trip(self):
        hidden = tensor((1, 1, 32), "float32")
        self.assertEqual(protocol.decode_forward_reply(protocol.encode_forward_reply(hidden)), hidden)

    def test_session_request_round_trip(self):
        self.assertEqual(protocol.decode_session_request(protocol.encode_session_request("abc")), "abc")

    def test_forward_request_rejects_bad_start_pos(self):
        body = protocol.encode_forward_request("s", tensor(), 0)
        for bad in (-1, 1.0, True, "3", None):
            with self.subTest(start_pos=bad):
                with self.assertRaises(ProtocolError):
                    protocol.decode_forward_request(dict(body, start_pos=bad))

    def test_forward_request_rejects_unknown_fields(self):
        body = protocol.encode_forward_request("s", tensor(), 0)
        with self.assertRaises(ProtocolError):
            protocol.decode_forward_request(dict(body, start=0))

    def test_session_ids(self):
        for good in ("a", "sess-1", "job_42.x", "x" * 128):
            self.assertEqual(protocol.check_session_id(good), good)
        for bad in ("", "../etc", "has space", "-leading", "x" * 129, None, 7):
            with self.subTest(session_id=bad):
                with self.assertRaises(ProtocolError):
                    protocol.check_session_id(bad)

    def test_encode_refuses_a_bad_session_id(self):
        with self.assertRaises(ProtocolError):
            protocol.encode_forward_request("bad id", tensor(), 0)

    def test_health_round_trip_and_validation(self):
        wire = protocol.encode_health((0, 8), "cpu")
        self.assertEqual(wire, {"ok": True, "layers": [0, 8], "device": "cpu"})
        self.assertEqual(protocol.decode_health(wire), ((0, 8), "cpu"))
        for bad in ((8, 8), (4, 2), (-1, 4), (0, 4.0)):
            with self.subTest(layers=bad):
                with self.assertRaises(ProtocolError):
                    protocol.check_layers(bad)

    def test_ok_reply(self):
        protocol.decode_ok(protocol.encode_ok())
        with self.assertRaises(ProtocolError):
            protocol.decode_ok({"ok": False})


if __name__ == "__main__":
    unittest.main()
