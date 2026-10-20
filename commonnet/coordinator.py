"""Coordinator: read member endpoints, check coverage, and generate through them.

    python -m pedal.coordinator --endpoints members.json --checkpoint DIR --prompt TEXT

--endpoints is a JSON file holding a list of {"url", "layers": [start, end], "token_file"},
or a comma-separated list of URL@START:END (token from PEDAL_BLOCK_TOKEN).
Prints the generated text only. It never prints the bearer token or token ids.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pedal.client import BlockEndpoint, Client
from pedal.net import auth
from pedal.net.transport import HttpTransport


class EndpointError(ValueError):
    pass


def load_endpoints(spec: str) -> list[dict]:
    if os.path.exists(spec):
        with open(spec) as f:
            raw = json.load(f)
    else:
        raw = []
        for part in spec.split(","):
            try:
                url, rng = part.strip().rsplit("@", 1)
                s, e = rng.split(":")
                raw.append({"url": url, "layers": [int(s), int(e)]})
            except ValueError:
                raise EndpointError(f"not a file, and {part!r} is not URL@START:END") from None
    if not isinstance(raw, list) or not raw:
        raise EndpointError("endpoints must be a non-empty JSON list")
    for i, m in enumerate(raw):
        if not isinstance(m, dict) or not isinstance(m.get("url"), str) or not m["url"] \
                or not isinstance(m.get("layers"), list) or len(m["layers"]) != 2 \
                or not all(type(x) is int for x in m["layers"]):
            raise EndpointError(f"member {i}: needs a url and layers [start, end]")
    return raw


def build_route(members: list[dict], num_layers: int):
    """Sort by start, refuse gaps, overlaps and a wrong total. Returns (endpoints, urls, tokens)."""
    ordered = sorted(members, key=lambda m: m["layers"][0])
    route = [BlockEndpoint(f"member{i}", m["layers"][0], m["layers"][1]) for i, m in enumerate(ordered)]
    try:
        Client.check_route(route, num_layers)
    except ValueError as e:
        raise EndpointError(f"layer coverage is wrong: {e}") from None
    urls, tokens = {}, {}
    for ep, m in zip(route, ordered):
        urls[ep.name] = m["url"]
        tokens[ep.name] = auth.load_token(m.get("token_file"))
        if tokens[ep.name] is None:
            raise EndpointError(f"no token for {ep.name}: give token_file or set {auth.TOKEN_ENV}")
    return route, urls, tokens


def generate(members, checkpoint, prompt_ids, max_new_tokens, fmt="fp32", transport_kwargs=None):
    from pedal.client4 import Client4
    from transformers import Qwen3Config
    n = Qwen3Config.from_pretrained(checkpoint).num_hidden_layers
    route, urls, tokens = build_route(members, n)
    transport = HttpTransport(urls, tokens=tokens, **(transport_kwargs or {}))
    client = Client4(checkpoint, transport, fmt=fmt)
    return client.generate(route, prompt_ids, max_new_tokens)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pedal.coordinator", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoints", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--client-format", choices=["fp32", "fp16", "mlx4"], default="fp16")
    args = p.parse_args(argv)
    try:
        from transformers import AutoTokenizer
        from pedal.net.client import BlockError
        members = load_endpoints(args.endpoints)
        tok = AutoTokenizer.from_pretrained(args.checkpoint)
        ids = tok(args.prompt)["input_ids"]
        out = generate(members, args.checkpoint, ids, args.max_new_tokens, args.client_format)
    except (EndpointError, auth.TokenError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except BlockError as e:
        print(f"error: member call failed: {e}", file=sys.stderr)
        return 1
    print(tok.decode(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
