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
