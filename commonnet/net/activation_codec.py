"""Per-position, per-group signed int8 activations for peer-to-peer links.

This is lossy quantization, not a claim of unchanged model quality. Token IDs and
model weights are untouched. Scales use little-endian float32; data uses int8.
"""
from __future__ import annotations

import base64

import numpy as np

from commonnet.net import protocol

CODEC = "int8_group_v1"
CAPABILITY = "activation_int8_group_v1"
GROUP_SIZE = 64
CODEC4 = "int4_group_v1"
CAPABILITY4 = "activation_int4_group_v1"
GROUP_SIZE4 = 32
CODEC_CAPABILITY = {CODEC: CAPABILITY, CODEC4: CAPABILITY4}
CODEC_PREFERENCE = (CODEC4, CODEC)   # most compressed first; "float" is the last resort
MAX_DECODED_BYTES = 4 * protocol.MAX_REQUEST_BYTES
_DTYPES = {"float16": np.dtype("<f2"), "float32": np.dtype("<f4")}
_FIELDS = ("codec", "shape", "dtype", "group_size", "data", "scales")


def _layout(shape, dtype):
    shape = protocol.check_shape(shape)
    dtype = protocol.check_dtype(dtype)
    count = protocol.numel(shape)
    if count * protocol.DTYPE_WIDTH[dtype] > MAX_DECODED_BYTES:
        raise protocol.ProtocolError("activation exceeds the decoded size limit")
    rows, width = shape[0] * shape[1], shape[2]
    groups = (width + GROUP_SIZE - 1) // GROUP_SIZE
    # Bound the base64 input before decoding or allocating any arrays. The outer
    # HTTP envelope retains its own stricter total request-size check.
    sizes = (count, rows * groups * 4)
    if sum(4 * ((n + 2) // 3) for n in sizes) > protocol.MAX_REQUEST_BYTES:
        raise protocol.ProtocolError("compressed activation exceeds the payload size limit")
    return shape, dtype, rows, width, groups, sizes


def _b64(raw):
    return base64.b64encode(raw).decode("ascii")


def _unb64(value, size, field):
    if not isinstance(value, str) or len(value) != 4 * ((size + 2) // 3):
        raise protocol.ProtocolError(f"activation {field} has the wrong encoded size")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, UnicodeError):
        raise protocol.ProtocolError(f"activation {field} is not valid base64") from None
    if len(raw) != size:
        raise protocol.ProtocolError(f"activation {field} has the wrong decoded size")
    return raw


def encode_int8_group(tensor: protocol.Tensor) -> dict:
    """Encode a float tensor with one symmetric scale per 64-wide group/position."""
    if not isinstance(tensor, protocol.Tensor):
        raise TypeError("encode_int8_group takes a protocol.Tensor")
    shape, dtype, rows, width, groups, _ = _layout(tensor.shape, tensor.dtype)
    values = np.frombuffer(tensor.data, dtype=_DTYPES[dtype]).reshape(rows, width).astype(np.float32)
    if not np.isfinite(values).all():
        raise protocol.ProtocolError("cannot compress non-finite activations")
    codes = np.empty((rows, width), dtype=np.int8)
    scales = np.empty((rows, groups), dtype="<f4")
    minimum = np.nextafter(np.float32(0), np.float32(1))
    # Group along hidden width independently for each batch/sequence position.
    # No padded tensor is allocated, including for a partial last group.
    for g, a in enumerate(range(0, width, GROUP_SIZE)):
        part = values[:, a:a + GROUP_SIZE]
        peak = np.max(np.abs(part), axis=1)
        scale = np.maximum(peak / np.float32(127), minimum)
        scale[peak == 0] = 1.0
        scales[:, g] = scale
        codes[:, a:a + GROUP_SIZE] = np.clip(np.rint(part / scale[:, None]), -127, 127).astype(np.int8)
    return {"codec": CODEC, "shape": list(shape), "dtype": dtype, "group_size": GROUP_SIZE,
            "data": _b64(codes.tobytes()), "scales": _b64(scales.tobytes())}


def decode_int8_group(obj) -> protocol.Tensor:
    """Strictly validate and restore a bounded int8 payload to its original dtype."""
    protocol._fields(obj, "compressed activation", _FIELDS)
    if obj["codec"] != CODEC or type(obj["group_size"]) is not int or obj["group_size"] != GROUP_SIZE:
        raise protocol.ProtocolError("unsupported activation codec or group size")
    if not isinstance(obj["shape"], list):
        raise protocol.ProtocolError("activation shape must be a JSON array")
    shape, dtype, rows, width, groups, sizes = _layout(obj["shape"], obj["dtype"])
    raw_codes = _unb64(obj["data"], sizes[0], "data")
    raw_scales = _unb64(obj["scales"], sizes[1], "scales")
    codes = np.frombuffer(raw_codes, dtype=np.int8).reshape(rows, width)
    if np.any(codes == -128):
        raise protocol.ProtocolError("activation codes must be between -127 and 127")
    scales = np.frombuffer(raw_scales, dtype="<f4").reshape(rows, groups)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise protocol.ProtocolError("activation scales must be finite and positive")
    values = np.empty((rows, width), dtype=_DTYPES[dtype])
    limit = np.finfo(_DTYPES[dtype]).max
    for g, a in enumerate(range(0, width, GROUP_SIZE)):
        # Float64 avoids overflow while multiplying a valid float32 extreme.
        restored = codes[:, a:a + GROUP_SIZE].astype(np.float64) * scales[:, g, None]
        values[:, a:a + GROUP_SIZE] = np.clip(restored, -limit, limit)
    return protocol.Tensor(shape, dtype, values.tobytes())


# ---- int4, group 32, float16 scales: two codes per byte. About 2.3 KB for a 4,096-wide position (float32: 16 KB,
# int8 above: 4.3 KB). Lossy: on Bonsai 8B at the layer-27 boundary it matched the unquantised greedy text on 7 of
# 12 prompts (int8: 11 of 12) and the top token 96.6% of the time (docs/24-oct-p&m-testing.md, sections 19-20).
_FIELDS4 = _FIELDS
_F16_MIN = np.float32(6.103515625e-05)   # smallest normal float16: a smaller scale would lose its precision
_F16_MAX = np.float32(65504.0)


def _layout4(shape, dtype):
    shape = protocol.check_shape(shape)
    dtype = protocol.check_dtype(dtype)
    count = protocol.numel(shape)
    if count * protocol.DTYPE_WIDTH[dtype] > MAX_DECODED_BYTES:
        raise protocol.ProtocolError("activation exceeds the decoded size limit")
    rows, width = shape[0] * shape[1], shape[2]
    groups = (width + GROUP_SIZE4 - 1) // GROUP_SIZE4
    padded = groups * GROUP_SIZE4          # the last group is zero-padded, so any hidden width works
    sizes = (rows * padded // 2, rows * groups * 2)
    if sum(4 * ((n + 2) // 3) for n in sizes) > protocol.MAX_REQUEST_BYTES:
        raise protocol.ProtocolError("compressed activation exceeds the payload size limit")
    return shape, dtype, rows, width, groups, padded, sizes


def encode_int4_group(tensor: protocol.Tensor) -> dict:
    """Symmetric int4 (-7..7), one float16 scale per 32 values of each position, two codes per byte."""
    if not isinstance(tensor, protocol.Tensor):
        raise TypeError("encode_int4_group takes a protocol.Tensor")
    shape, dtype, rows, width, groups, padded, _ = _layout4(tensor.shape, tensor.dtype)
    values = np.zeros((rows, padded), dtype=np.float32)
    values[:, :width] = np.frombuffer(tensor.data, dtype=_DTYPES[dtype]).reshape(rows, width)
    if not np.isfinite(values).all():
        raise protocol.ProtocolError("cannot compress non-finite activations")
    values = values.reshape(rows, groups, GROUP_SIZE4)
    peak = np.max(np.abs(values), axis=2)
    scale16 = np.clip(peak / np.float32(7), _F16_MIN, _F16_MAX).astype("<f2")
    codes = np.clip(np.rint(values / scale16.astype(np.float32)[:, :, None]), -7, 7).astype(np.int8).reshape(rows, padded)
    nib = (codes & 0x0F).astype(np.uint8)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return {"codec": CODEC4, "shape": list(shape), "dtype": dtype, "group_size": GROUP_SIZE4,
            "data": _b64(packed.tobytes()), "scales": _b64(scale16.tobytes())}


def decode_int4_group(obj) -> protocol.Tensor:
    """Strictly validate and restore a bounded int4 payload to its original dtype."""
    protocol._fields(obj, "compressed activation", _FIELDS4)
    if obj["codec"] != CODEC4 or type(obj["group_size"]) is not int or obj["group_size"] != GROUP_SIZE4:
        raise protocol.ProtocolError("unsupported activation codec or group size")
    if not isinstance(obj["shape"], list):
        raise protocol.ProtocolError("activation shape must be a JSON array")
    shape, dtype, rows, width, groups, padded, sizes = _layout4(obj["shape"], obj["dtype"])
    raw_codes = _unb64(obj["data"], sizes[0], "data")
    raw_scales = _unb64(obj["scales"], sizes[1], "scales")
    packed = np.frombuffer(raw_codes, dtype=np.uint8).reshape(rows, padded // 2)
    codes = np.empty((rows, padded), dtype=np.int8)
    codes[:, 0::2], codes[:, 1::2] = (packed & 0x0F).astype(np.int8), (packed >> 4).astype(np.int8)
    codes = np.where(codes > 7, codes - 16, codes)
    if np.any(codes == -8):
        raise protocol.ProtocolError("activation codes must be between -7 and 7")
    scales = np.frombuffer(raw_scales, dtype="<f2").astype(np.float32).reshape(rows, groups)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise protocol.ProtocolError("activation scales must be finite and positive")
    restored = codes.reshape(rows, groups, GROUP_SIZE4).astype(np.float64) * scales[:, :, None]
    limit = np.finfo(_DTYPES[dtype]).max
    values = np.clip(restored, -limit, limit).astype(_DTYPES[dtype]).reshape(rows, padded)[:, :width]
    return protocol.Tensor(shape, dtype, np.ascontiguousarray(values).tobytes())


def encode_activation(codec: str, tensor: protocol.Tensor):
    """The wire form of a hidden state for `codec`: a compressed dict, or the plain tensor encoding for float."""
    if codec == CODEC4:
        return encode_int4_group(tensor)
    if codec == CODEC:
        return encode_int8_group(tensor)
    return protocol.encode_tensor(tensor)


def decode_activation(codec: str, obj) -> protocol.Tensor:
    if codec == CODEC4:
        return decode_int4_group(obj)
    if codec == CODEC:
        return decode_int8_group(obj)
    return protocol.decode_tensor(obj)
