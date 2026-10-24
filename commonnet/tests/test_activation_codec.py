"""Basic int8 activation wire checks; no model-quality comparison."""
import base64
import unittest

import numpy as np

from commonnet.net import activation_codec as codec, protocol


class ActivationCodec(unittest.TestCase):
    def tensor(self, shape=(1, 2, 70), dtype="float32"):
        values = np.arange(protocol.numel(shape), dtype=np.float32).reshape(shape) - 60
        return protocol.Tensor(shape, dtype, values.astype("<f4" if dtype == "float32" else "<f2").tobytes())

    def test_smoke_preserves_layout_and_compacts_payload(self):
        for dtype in ("float16", "float32"):
            with self.subTest(dtype=dtype):
                original = self.tensor(dtype=dtype)
                encoded = codec.encode_int8_group(original)
                restored = codec.decode_int8_group(encoded)
                self.assertEqual(restored.shape, original.shape)
                self.assertEqual(restored.dtype, dtype)
                self.assertEqual(len(restored.data), len(original.data))
                self.assertTrue(np.isfinite(np.frombuffer(restored.data, dtype="<f4" if dtype == "float32" else "<f2")).all())
                self.assertEqual(len(base64.b64decode(encoded["data"])), protocol.numel(original.shape))

    def test_zero_groups_are_safe(self):
        original = protocol.Tensor((1, 1, 65), "float32", bytes(65 * 4))
        self.assertEqual(codec.decode_int8_group(codec.encode_int8_group(original)), original)

    def test_bad_wire_shapes_lengths_fields_and_scales_are_rejected(self):
        good = codec.encode_int8_group(self.tensor())
        malformed = [dict(good, codec="unknown"), dict(good, group_size=True),
                     dict(good, shape=[1, 2, 71]), dict(good, data="!" * len(good["data"])),
                     dict(good, scales=""), dict(good, extra=1),
                     dict(good, shape=[1, 1, 2**40])]
        for scale in (0.0, -1.0, float("inf"), float("nan")):
            raw = np.full(4, scale, dtype="<f4").tobytes()
            malformed.append(dict(good, scales=base64.b64encode(raw).decode()))
        for obj in malformed:
            with self.subTest(obj={k: v for k, v in obj.items() if k not in ("data", "scales")}), self.assertRaises(protocol.ProtocolError):
                codec.decode_int8_group(obj)

    def test_nonfinite_input_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            original = protocol.Tensor((1, 1, 1), "float32", np.array([value], dtype="<f4").tobytes())
            with self.assertRaises(protocol.ProtocolError):
                codec.encode_int8_group(original)


if __name__ == "__main__":
    unittest.main()
