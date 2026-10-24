"""One synthetic node as its own process: the real commonnet Node (gossip, placement, signed record, BlockServer)
serving a SynthBlock. Prints READY once it is serving. Stops on SIGTERM.

    python -m commonnet.bench.nodeproc --manifest M.json --port P --advertise URL --identity-dir D [--seed URL]
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from commonnet.mesh.manifest import Manifest
from commonnet.mesh.records import Identity
from commonnet.node import Hardware, Node

from commonnet.bench.synth import DEFAULT_SEED, SynthBlock


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--advertise", help="plain mode: the URL to publish (gossip only, as a tunnel peer would)")
    p.add_argument("--direct-loopback", action="store_true", help="direct mode in test form: TLS on 127.0.0.1, found through the DHT")
    p.add_argument("--bootstrap", action="append", default=[], help="with --direct-loopback: DHT bootstrap HOST:PORT")
    p.add_argument("--identity-dir", required=True)
    p.add_argument("--seed", action="append", default=[])
    p.add_argument("--window", help="serve exactly this window, A-B (default: whatever the network needs)")
    p.add_argument("--layer-ms", type=float, default=0.0)
    p.add_argument("--weights-seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--stage-layers", type=int, required=True, help="layers per stage; the node offers exactly this many GiB")
    args = p.parse_args(argv)
    manifest = Manifest.load(args.manifest)
    identity = Identity.load_or_create(Path(args.identity_dir) / "identity.key")
    hardware = Hardware("synth", float(args.stage_layers), "synthetic")

    def build(window, _manifest, _hw):
        return SynthBlock(window[0], window[1], layer_ms=args.layer_ms, seed=args.weights_seed)

    fixed = tuple(int(x) for x in args.window.split("-")) if args.window else None
    direct = None
    if args.direct_loopback:
        from commonnet.direct import DirectConfig
        direct = DirectConfig(bootstrap=args.bootstrap, loopback=True, tunnel_fallback=False)
    elif not args.advertise:
        p.error("--advertise or --direct-loopback is required")
    node = Node(manifest, args.seed, identity, hardware, port=args.port, tunnel=False, advertise=args.advertise,
                block_builder=build, log=lambda m: None, gossip_s=1.0, direct=direct, fixed_window=fixed,
                state_dir=Path(args.identity_dir) / "tls", resign_s=2.0 if direct else 15.0)
    signal.signal(signal.SIGTERM, lambda *_: node.request_leave())
    signal.signal(signal.SIGINT, lambda *_: node.request_leave())
    try:
        node.start()
        print("READY", node.window[0], node.window[1], flush=True)
        node.wait()
    finally:
        node.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
