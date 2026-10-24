"""Wire format for the block server: tensors as base64 inside JSON.

A tensor travels as {"shape": [batch, seq, hidden], "dtype": "float16", "data": "<base64>"}.
The data is the tensor's raw bytes in C order, little-endian. A 4096-wide float16
token is 8192 bytes, which is about 11 KB of base64.

Everything here is strict. A tensor has exactly three dimensions, each a positive
int (a bool, a float or a string is refused). The dtype is one of the names in
DTYPE_WIDTH. The base64 must decode strictly, and the byte count must equal the
shape times the dtype's width. Envelopes refuse fields they do not know, so a
misspelt field fails loudly instead of being ignored.

This module knows nothing about torch or numpy. The block object exchanges Tensor
values, and converting to and from the model's own tensor type happens at the
block's edge.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass

DTYPE_WIDTH = {"float16": 2, "float32": 4}
SHAPE_RANK = 3
# A request body larger than this is refused with 413. 8 MiB holds about 700 tokens
# of prefill at hidden size 4096 in float16, so longer prompts must be sent in pieces.
MAX_REQUEST_BYTES = 8 * 1024 * 1024
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class ProtocolError(ValueError):
    """A message or tensor that breaks the wire format."""


def numel(shape: tuple) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def check_shape(shape) -> tuple:
    if not isinstance(shape, (list, tuple)) or len(shape) != SHAPE_RANK:
        raise ProtocolError(f"shape must be [batch, seq, hidden], got {shape!r}")
    for dim in shape:
        if type(dim) is not int or dim < 1:
            raise ProtocolError(f"every shape dimension must be a positive int, got {shape!r}")
    return tuple(shape)


def check_dtype(dtype) -> str:
    if not isinstance(dtype, str) or dtype not in DTYPE_WIDTH:
        raise ProtocolError(f"dtype must be one of {sorted(DTYPE_WIDTH)}, got {dtype!r}")
    return dtype


def check_session_id(value) -> str:
    if not isinstance(value, str) or SESSION_ID_PATTERN.fullmatch(value) is None:
        raise ProtocolError("session id must be 1 to 128 characters: letters, digits, '_', '.' or '-'")
    return value


def check_start_pos(value) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolError(f"start_pos must be a non-negative int, got {value!r}")
    return value


def check_layers(value) -> tuple:
    """A half-open layer range [start, end): 0 <= start < end."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ProtocolError(f"layers must be [start, end], got {value!r}")
    start, end = value
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ProtocolError(f"layers must satisfy 0 <= start < end, got {value!r}")
    return (start, end)


@dataclass(frozen=True)
class Tensor:
    """A tensor on the wire: raw bytes with their shape and dtype.

    Construction validates everything, so a Tensor that exists is always
    internally consistent.
    """

    shape: tuple
    dtype: str
    data: bytes

    def __post_init__(self) -> None:
        shape = check_shape(self.shape)
        dtype = check_dtype(self.dtype)
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise ProtocolError("tensor data must be bytes")
        data = bytes(self.data)
        expected = numel(shape) * DTYPE_WIDTH[dtype]
        if len(data) != expected:
            raise ProtocolError(
                f"tensor data is {len(data)} bytes, but shape {list(shape)} {dtype} needs {expected}"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "data", data)


def _fields(obj, what: str, required: tuple, optional: tuple = ()) -> None:
    if not isinstance(obj, dict):
        raise ProtocolError(f"{what} must be a JSON object")
    missing = [key for key in required if key not in obj]
    if missing:
        raise ProtocolError(f"{what} is missing {', '.join(missing)}")
    unknown = [key for key in obj if key not in required and key not in optional]
    if unknown:
        raise ProtocolError(f"{what} has unknown fields {', '.join(map(str, unknown))}")


def encode_tensor(tensor: Tensor) -> dict:
    if not isinstance(tensor, Tensor):
        raise TypeError("encode_tensor takes a protocol.Tensor")
    return {
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
        "data": base64.b64encode(tensor.data).decode("ascii"),
    }


def decode_tensor(obj, what: str = "tensor") -> Tensor:
    _fields(obj, what, ("shape", "dtype", "data"))
    if not isinstance(obj["shape"], list):
        raise ProtocolError(f"{what} shape must be a JSON array")
    if not isinstance(obj["data"], str):
        raise ProtocolError(f"{what} data must be a base64 string")
    try:
        raw = base64.b64decode(obj["data"], validate=True)
    except ValueError:  # binascii.Error is a ValueError
        raise ProtocolError(f"{what} data is not valid base64") from None
    return Tensor(tuple(obj["shape"]), obj["dtype"], raw)


def encode_session_request(session_id: str) -> dict:
    return {"id": check_session_id(session_id)}


def decode_session_request(obj) -> str:
    _fields(obj, "session request", ("id",))
    return check_session_id(obj["id"])


def encode_truncate_request(session_id: str, length: int) -> dict:
    if type(length) is not int or length < 0:
        raise ProtocolError(f"length must be a non-negative int, got {length!r}")
    return {"id": check_session_id(session_id), "length": length}


def decode_truncate_request(obj) -> tuple:
    """Returns (session_id, length)."""
    _fields(obj, "truncate request", ("id", "length"))
    session_id = check_session_id(obj["id"])
    length = obj["length"]
    if type(length) is not int or length < 0:
        raise ProtocolError(f"length must be a non-negative int, got {length!r}")
    return session_id, length


def encode_forward_request(session_id: str, hidden: Tensor, start_pos: int) -> dict:
    body = {"id": check_session_id(session_id), "start_pos": check_start_pos(start_pos)}
    body.update(encode_tensor(hidden))
    return body


def decode_forward_request(obj) -> tuple:
    """Returns (session_id, start_pos, hidden)."""
    _fields(obj, "forward request", ("id", "start_pos", "shape", "dtype", "data"))
    session_id = check_session_id(obj["id"])
    start_pos = check_start_pos(obj["start_pos"])
    hidden = decode_tensor({key: obj[key] for key in ("shape", "dtype", "data")}, "forward request")
    return session_id, start_pos, hidden


def encode_forward_reply(hidden: Tensor) -> dict:
    return encode_tensor(hidden)


def decode_forward_reply(obj) -> Tensor:
    return decode_tensor(obj, "forward reply")


def encode_ok() -> dict:
    return {"ok": True}


def decode_ok(obj) -> None:
    _fields(obj, "reply", ("ok",))
    if obj["ok"] is not True:
        raise ProtocolError("reply does not say ok")


def encode_error(message: str) -> dict:
    return {"ok": False, "error": message}


def encode_health(layers, device: str) -> dict:
    return {"ok": True, "layers": list(check_layers(layers)), "device": str(device)}


def decode_health(obj) -> tuple:
    """Returns (layers, device)."""
    _fields(obj, "health reply", ("ok", "layers", "device"))
    if obj["ok"] is not True:
        raise ProtocolError("health reply does not say ok")
    if not isinstance(obj["device"], str):
        raise ProtocolError("health reply device must be a string")
    return check_layers(obj["layers"]), obj["device"]
