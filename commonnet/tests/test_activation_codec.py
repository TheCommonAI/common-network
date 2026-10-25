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



class Int4ActivationCodec(unittest.TestCase):
    def activation(self, shape=(1, 3, 4096), dtype="float32", seed=0):
        rng = np.random.default_rng(seed)
        values = (rng.standard_normal(shape) * 5).astype("<f4")
        values[0, 0, 7] = 900.0  # an outlier channel, as in real hidden states
        raw = values.astype("<f4" if dtype == "float32" else "<f2")
        return values, protocol.Tensor(shape, dtype, raw.tobytes())

    def test_round_trip_keeps_layout_and_error_stays_within_half_a_step_per_group(self):
        for dtype in ("float16", "float32"):
            with self.subTest(dtype=dtype):
                values, tensor = self.activation(dtype=dtype)
                restored = codec.decode_int4_group(codec.encode_int4_group(tensor))
                self.assertEqual((restored.shape, restored.dtype, len(restored.data)), (tensor.shape, dtype, len(tensor.data)))
                got = np.frombuffer(restored.data, dtype="<f4" if dtype == "float32" else "<f2").astype(np.float32).reshape(1, 3, 128, 32)
                want = np.frombuffer(tensor.data, dtype="<f4" if dtype == "float32" else "<f2").astype(np.float32).reshape(1, 3, 128, 32)
                step = np.abs(want).max(-1, keepdims=True) / 7
                self.assertTrue((np.abs(got - want) <= step * 0.51 + 1e-3).all())

    def test_size_is_about_half_a_byte_per_value_plus_a_float16_scale_per_32(self):
        _, tensor = self.activation()
        encoded = codec.encode_int4_group(tensor)
        self.assertEqual(len(base64.b64decode(encoded["data"])), 3 * 4096 // 2)
        self.assertEqual(len(base64.b64decode(encoded["scales"])), 3 * 128 * 2)

    def test_decoded_values_are_the_tested_quantiser(self):
        # same numerics as the quality experiments: symmetric, scale = peak / 7 rounded to float16, group 32
        values, tensor = self.activation()
        got = np.frombuffer(codec.decode_int4_group(codec.encode_int4_group(tensor)).data, dtype="<f4").reshape(1, 3, 128, 32)
        g = values.reshape(1, 3, 128, 32)
        scale = (np.abs(g).max(-1, keepdims=True) / 7).astype(np.float16).astype(np.float32)
        want = np.clip(np.rint(g / scale), -7, 7) * scale
        self.assertTrue(np.allclose(got, want, rtol=0, atol=1e-5))

    def test_zero_group_and_non_finite_values(self):
        zeros = protocol.Tensor((1, 1, 64), "float32", np.zeros(64, dtype="<f4").tobytes())
        self.assertTrue((np.frombuffer(codec.decode_int4_group(codec.encode_int4_group(zeros)).data, dtype="<f4") == 0).all())
        bad = protocol.Tensor((1, 1, 64), "float32", np.full(64, np.inf, dtype="<f4").tobytes())
        with self.assertRaises(protocol.ProtocolError):
            codec.encode_int4_group(bad)

    def test_any_hidden_width_works_with_a_padded_last_group(self):
        for width in (1, 7, 33, 70, 4097):
            with self.subTest(width=width):
                x = (np.arange(2 * width, dtype=np.float32) - width).astype("<f4")
                tensor = protocol.Tensor((1, 2, width), "float32", x.tobytes())
                restored = codec.decode_int4_group(codec.encode_int4_group(tensor))
                got = np.frombuffer(restored.data, dtype="<f4")
                self.assertEqual((restored.shape, len(got)), ((1, 2, width), 2 * width))
                self.assertTrue(np.abs(got - x).max() <= np.abs(x).max() / 7 * 0.51 + 1e-3)

    def test_malformed_payloads_are_refused(self):
        _, tensor = self.activation()
        good = codec.encode_int4_group(tensor)
        for change in ({"codec": "int8_group_v1"}, {"group_size": 64}, {"data": good["data"][:-4]}, {"scales": "AAAA"},
                       {"shape": "x"}):
            with self.subTest(change=list(change)):
                with self.assertRaises(protocol.ProtocolError):
                    codec.decode_int4_group({**good, **change})
        bad = bytearray(base64.b64decode(good["data"])); bad[0] = (bad[0] & 0xF0) | 0x08  # code -8 is not produced
        with self.assertRaises(protocol.ProtocolError):
            codec.decode_int4_group({**good, "data": base64.b64encode(bytes(bad)).decode()})
        zero_scale = {**good, "scales": base64.b64encode(bytes(len(base64.b64decode(good["scales"])))).decode()}
        with self.assertRaises(protocol.ProtocolError):
            codec.decode_int4_group(zero_scale)

    def test_dispatch_helpers_and_capabilities(self):
        _, tensor = self.activation()
        for name in ("float", codec.CODEC, codec.CODEC4):
            with self.subTest(codec=name):
                restored = codec.decode_activation(name, codec.encode_activation(name, tensor))
                self.assertEqual((restored.shape, restored.dtype), (tensor.shape, tensor.dtype))
        self.assertEqual(codec.CODEC_CAPABILITY[codec.CODEC4], "activation_int4_group_v1")
        self.assertIn("activation_int4_group_v1", protocol.CAPS)


class SessionCodecChoice(unittest.TestCase):
    """The transport uses the most compressed codec every hop supports, never more than it was asked for."""

    def choose(self, preferred, caps_per_hop):
        from commonnet.net.transport import HttpTransport
        t = HttpTransport.__new__(HttpTransport)
        t._activation_codec = preferred
        return t._session_codec([{"caps": caps} for caps in caps_per_hop])

    def test_int4_when_asked_for_falls_back_to_int8_then_float(self):
        both = ("activation_int8_group_v1", "activation_int4_group_v1")
        old = ("activation_int8_group_v1",)
        self.assertEqual(self.choose("int4_group_v1", [both, both]), "int4_group_v1")
        self.assertEqual(self.choose("int4_group_v1", [both, old]), "int8_group_v1")
        self.assertEqual(self.choose("int4_group_v1", [both, ()]), "float")
        self.assertEqual(self.choose("int8_group_v1", [both, both]), "int8_group_v1")  # the default never goes up to int4
        self.assertEqual(self.choose("int8_group_v1", [old, old]), "int8_group_v1")
        self.assertEqual(self.choose("int8_group_v1", [old, ()]), "float")
        self.assertEqual(self.choose("float", [both, both]), "float")

    def test_env_and_argument_select_the_preferred_codec(self):
        import os
        from unittest import mock
        from commonnet.net.transport import HttpTransport
        urls = {"s0": "http://127.0.0.1:9"}
        for value, want in (("int4_group", "int4_group_v1"), ("int8_group", "int8_group_v1"), ("float", "float")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"COMMONNET_ACTIVATION_CODEC": value}):
                self.assertEqual(HttpTransport(urls, token="t" * 32)._activation_codec, want)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COMMONNET_ACTIVATION_CODEC", None)
            self.assertEqual(HttpTransport(urls, token="t" * 32)._activation_codec, "int8_group_v1")  # the default stays int8
            self.assertEqual(HttpTransport(urls, token="t" * 32, activation_codec="int4_group")._activation_codec, "int4_group_v1")
            self.assertEqual(HttpTransport(urls, token="t" * 32, activation_codec="int8_group")._activation_codec, "int8_group_v1")
        with mock.patch.dict(os.environ, {"COMMONNET_ACTIVATION_CODEC": "nonsense"}):
            with self.assertRaises(ValueError):
                HttpTransport(urls, token="t" * 32)


if __name__ == "__main__":
    unittest.main()
