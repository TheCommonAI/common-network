"""python -m commonnet.dht --bootstrap HOST:PORT [--network ID] [--list]"""
from __future__ import annotations

import argparse
import sys
import time

from ..mesh.records import Identity
from .bootstrap import load_bootstrap, parse_addr
from .node import DhtNode

PRINT_FIELDS = ("peer", "url", "backend", "state", "gib", "window", "seq", "ts")  # never the token


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m commonnet.dht", description="Run a plain DHT node (a bootstrap node).")
    ap.add_argument("--bootstrap", action="append", default=[], metavar="HOST:PORT",
                    help="a node to join through (repeatable); also commonnet/networks/bootstrap.json and COMMONNET_DHT_BOOTSTRAP")
    ap.add_argument("--network", default="", help="network id (needed for --list)")
    ap.add_argument("--bind", default="0.0.0.0:0", metavar="HOST:PORT", help="UDP address to bind (default any port)")
    ap.add_argument("--list", action="store_true", help="join, print the live peers of --network, and exit")
    args = ap.parse_args(argv)
    boot = [parse_addr(a) for a in args.bootstrap] + [a for a in load_bootstrap() if a not in args.bootstrap]
    node = DhtNode(Identity.generate(), bind=parse_addr(args.bind, allow_zero_port=True), bootstrap=boot, network_id=args.network)
    node.start()
    try:
        if args.list:
            if not args.network:
                print("--list needs --network", file=sys.stderr)
                return 2
            node.bootstrap()
            for rec in node.find_peers():
                print(" ".join(f"{k}={rec[k]}" for k in PRINT_FIELDS))
            return 0
        print(f"dht node {node.peer} listening on udp {node.address[0]}:{node.address[1]}", flush=True)
        while True:
            time.sleep(60)
            print(f"contacts={len(node.table)} stored={node.store.count()}", flush=True)
    except KeyboardInterrupt:
        return 0
    finally:
        node.stop()


if __name__ == "__main__":
    sys.exit(main())
