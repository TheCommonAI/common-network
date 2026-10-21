"""Start a member: the block server plus its tunnel, and write the endpoint JSON.

    COMMONNET_BLOCK_TOKEN=... python -m commonnet.member --layers 0 28 --checkpoint DIR --out member.json
    ... --repo unsloth/Qwen3-8B-bnb-4bit --layers 20 36   (downloads only those layers)

The token comes from COMMONNET_BLOCK_TOKEN or --token-file (never from the command line).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from commonnet.net import auth, factory, tunnel


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="commonnet.member", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", nargs=2, type=int, metavar=("START", "END"), required=True)
    p.add_argument("--checkpoint", help="local checkpoint directory (skips the download)")
    p.add_argument("--repo", help="Hugging Face repo id; downloads only --layers (HF_TOKEN for gated repos)")
    p.add_argument("--revision", default="main")
    p.add_argument("--backend", choices=factory.BACKENDS, default=None,
                   help="torch (default), mlx, or nf4 (CUDA only); or set COMMONNET_BACKEND")
    p.add_argument("--token-file", metavar="PATH")
    p.add_argument("--out", type=Path, default=Path("member.json"))
    args = p.parse_args(argv)
    try:
        token = auth.load_token(args.token_file)
    except auth.TokenError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if token is None:
        print(f"error: set {auth.TOKEN_ENV} or pass --token-file", file=sys.stderr)
        return 2
    if not args.checkpoint:
        if not args.repo:
            print("error: give --checkpoint DIR or --repo REPO", file=sys.stderr)
            return 2
        from commonnet import hub
        try:
            args.checkpoint = hub.Hub(args.repo, args.revision).fetch(layers=tuple(args.layers))
        except hub.HubError as e:
            print(f"error: download failed: {e}", file=sys.stderr)
            return 1
    os.environ["COMMONNET_CHECKPOINT"] = args.checkpoint
    if args.backend:
        os.environ["COMMONNET_BACKEND"] = args.backend
    targv = ["--layers", str(args.layers[0]), str(args.layers[1]),
             "--block", "commonnet.net.factory:FACTORY", "--out", str(args.out)]
    if args.token_file:
        targv += ["--token-file", args.token_file]
    return tunnel.main(targv)


if __name__ == "__main__":
    sys.exit(main())
