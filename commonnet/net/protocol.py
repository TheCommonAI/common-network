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
import math
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


def encode_forward_request(session_id: str, hidden: Tensor, start_pos: int, truncate_to: int | None = None) -> dict:
    """truncate_to (optional): cut the session back to that many positions before running this step, so a speculative
    rollback rides on the next forward instead of costing a round trip of its own. Old servers refuse the field."""
    body = {"id": check_session_id(session_id), "start_pos": check_start_pos(start_pos)}
    if truncate_to is not None:
        body["truncate_to"] = check_start_pos(truncate_to)
    body.update(encode_tensor(hidden))
    return body


def decode_forward_request(obj) -> tuple:
    """Returns (session_id, start_pos, hidden)."""
    _fields(obj, "forward request", ("id", "start_pos", "shape", "dtype", "data"))
    session_id = check_session_id(obj["id"])
    start_pos = check_start_pos(obj["start_pos"])
    hidden = decode_tensor({key: obj[key] for key in ("shape", "dtype", "data")}, "forward request")
    return session_id, start_pos, hidden


def decode_forward_request_full(obj) -> tuple:
    """Returns (session_id, start_pos, hidden, truncate_to or None)."""
    _fields(obj, "forward request", ("id", "start_pos", "shape", "dtype", "data"), ("truncate_to",))
    session_id, start_pos = check_session_id(obj["id"]), check_start_pos(obj["start_pos"])
    truncate_to = check_start_pos(obj["truncate_to"]) if "truncate_to" in obj else None
    hidden = decode_tensor({key: obj[key] for key in ("shape", "dtype", "data")}, "forward request")
    return session_id, start_pos, hidden, truncate_to


MAX_IDS = 1 << 20
OUTPUTS = ("ids",)
WHICH = ("last", "all")
CAPS = ("ids_in", "head_out", "chain_v1", "chain_private_v1", "activation_int8_group_v1", "activation_int4_group_v1")


def check_ids(value) -> list:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_IDS:
        raise ProtocolError("ids must be a non-empty JSON array of token ids")
    for t in value:
        if type(t) is not int or t < 0:
            raise ProtocolError("every token id must be a non-negative int")
    return value


def check_sampling(value):
    """None (greedy) or {"temperature": >0, "top_p": in (0, 1], "seed": int >= 0}."""
    if value is None:
        return None
    _fields(value, "sampling", ("temperature", "top_p", "seed"))
    t, p, seed = value["temperature"], value["top_p"], value["seed"]
    if type(t) not in (int, float) or type(p) not in (int, float):
        raise ProtocolError("sampling needs temperature > 0 and 0 < top_p <= 1")
    try:
        t, p = float(t), float(p)
    except (ValueError, OverflowError):
        raise ProtocolError("sampling needs finite temperature and top_p") from None
    if not math.isfinite(t) or not t > 0 or not math.isfinite(p) or not 0 < p <= 1:
        raise ProtocolError("sampling needs temperature > 0 and 0 < top_p <= 1")
    if type(seed) is not int or seed < 0:
        raise ProtocolError("sampling seed must be a non-negative int")
    return {"temperature": float(t), "top_p": float(p), "seed": seed}


@dataclass(frozen=True)
class ForwardRequest:
    """A /forward body: exactly one of hidden / ids, plus the optional output mode ("ids": return token ids)."""
    session_id: str
    start_pos: int
    hidden: Tensor | None = None
    ids: list | None = None
    truncate_to: int | None = None
    output: str | None = None
    which: str = "last"
    sampling: dict | None = None


def encode_forward_request_ex(session_id: str, hidden: Tensor | None, start_pos: int, truncate_to: int | None = None, *,
                              ids=None, output: str | None = None, which: str | None = None, sampling=None) -> dict:
    """Like encode_forward_request, but the input may be token ids (hidden=None) and the reply token ids
    (output="ids"; which="last" or "all" positions; sampling None = greedy). Old servers refuse the new fields."""
    if (hidden is None) == (ids is None):
        raise ProtocolError("a forward request carries hidden or ids, not both and not neither")
    body = {"id": check_session_id(session_id), "start_pos": check_start_pos(start_pos)}
    if truncate_to is not None:
        body["truncate_to"] = check_start_pos(truncate_to)
    if ids is not None:
        body["ids"] = check_ids(list(ids))
    else:
        body.update(encode_tensor(hidden))
    if output is not None:
        if output not in OUTPUTS:
            raise ProtocolError(f"output must be one of {OUTPUTS}")
        body["output"] = output
        body["which"] = which or "last"
        if sampling is not None:
            body["sampling"] = check_sampling(sampling)
    return body


def decode_forward_request_ex(obj) -> ForwardRequest:
    if isinstance(obj, dict) and "ids" in obj:
        _fields(obj, "forward request", ("id", "start_pos", "ids"), ("truncate_to", "output", "which", "sampling"))
        hidden, ids = None, check_ids(obj["ids"])
    else:
        _fields(obj, "forward request", ("id", "start_pos", "shape", "dtype", "data"),
                ("truncate_to", "output", "which", "sampling"))
        hidden, ids = decode_tensor({k: obj[k] for k in ("shape", "dtype", "data")}, "forward request"), None
    output = obj.get("output")
    if output is not None and output not in OUTPUTS:
        raise ProtocolError(f"output must be one of {OUTPUTS}")
    which = obj.get("which", "last")
    if which not in WHICH or (("which" in obj or "sampling" in obj) and output is None):
        raise ProtocolError("which and sampling need output")
    return ForwardRequest(check_session_id(obj["id"]), check_start_pos(obj["start_pos"]), hidden, ids,
                          check_start_pos(obj["truncate_to"]) if "truncate_to" in obj else None, output, which,
                          check_sampling(obj.get("sampling")))


def encode_ids_reply(ids) -> dict:
    return {"ids": check_ids(list(ids))}


def decode_ids_reply(obj) -> list:
    _fields(obj, "forward reply", ("ids",))
    return check_ids(obj["ids"])


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


def encode_health(layers, device: str, caps=None) -> dict:
    """caps (optional): what this block can do beyond the basic protocol. Sent only to a client that asked
    (GET /health?caps=1), because an old client refuses health replies with fields it does not know."""
    out = {"ok": True, "layers": list(check_layers(layers)), "device": str(device)}
    if caps:
        out["caps"] = [c for c in caps if c in CAPS]
    return out


def decode_health_caps(obj) -> tuple:
    """The capability names in a health reply (empty from an old server)."""
    caps = obj.get("caps", []) if isinstance(obj, dict) else []
    return tuple(c for c in caps if isinstance(c, str)) if isinstance(caps, list) else ()


def decode_health(obj) -> tuple:
    """Returns (layers, device)."""
    _fields(obj, "health reply", ("ok", "layers", "device"), ("caps",))
    if obj["ok"] is not True:
        raise ProtocolError("health reply does not say ok")
    if not isinstance(obj["device"], str):
        raise ProtocolError("health reply device must be a string")
    return check_layers(obj["layers"]), obj["device"]
