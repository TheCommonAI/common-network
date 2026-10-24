"""Direct mode (experimental): how a node decides the address it publishes when it serves without a tunnel.

    python -m commonnet.node --direct --bootstrap HOST:PORT --port 7471 --public-ip 203.0.113.9 --external-port 7471

Everything here is off unless `--direct` is given. `decide()` is pure apart from the functions passed in (the
port mapper, STUN, the dial-back), so every branch is tested with fakes; nothing in the tests touches a router.

The order of preference, and what each one means:

  manual    --public-ip and/or --external-port: the owner forwarded a port by hand. First class, because many
            routers offer no UPnP/PCP/NAT-PMP. Still checked by a remote dial-back when anyone can be asked.
  mapped    the router opened a port (UPnP, then PCP, then NAT-PMP). Believed only after a dial-back succeeds.
  stun      no mapping: the public IP STUN reports plus the local port. Works when this machine has a public address
            or the port is already forwarded. Believed only after a dial-back succeeds.
  loopback  tests only (--direct-loopback): 127.0.0.1, TLS and pinning for real, no reachability check.

If the dial-back says nobody can connect, the verdict is `unreachable` and the node falls back to its Cloudflare
tunnel (or refuses, with --no-tunnel-fallback). A private, loopback or link-local address is never published.
"""
from __future__ import annotations

import dataclasses
import ipaddress

from commonnet.net import nat

LOOPBACK_IP = "127.0.0.1"
DEFAULT_TCP_PORT = 7471    # the block server in auto mode
DEFAULT_DHT_PORT = 7472    # the DHT (UDP) in auto mode


@dataclasses.dataclass
class DirectConfig:
    bootstrap: list = dataclasses.field(default_factory=list)   # extra "HOST:PORT" DHT bootstrap nodes
    external_port: int | None = None     # the TCP port the outside world uses (manual forward)
    public_ip: str | None = None         # this machine's public IPv4 (manual forward)
    dht_port: int = 0                    # UDP port of the DHT node (0 = any; forward it for others to reach you)
    tunnel_fallback: bool = True
    auto: bool = False                   # plain `common join`: default ports 7471/7472 (free-port fallback), tunnel on failure
    loopback: bool = False               # tests only: publish https://127.0.0.1:PORT, skip reachability


@dataclasses.dataclass
class Decision:
    kind: str                    # manual | mapped | stun | loopback | unreachable
    ip: str | None = None
    port: int | None = None
    verified: bool | None = None  # True: a remote peer connected. None: nobody could be asked. False: refused
    detail: str = ""
    mapping: object = None       # a nat.Mapping to release on shutdown

    @property
    def direct(self) -> bool:
        return self.kind != "unreachable"

    @property
    def url(self) -> str | None:
        return f"https://{self.ip}:{self.port}" if self.direct else None


def is_public_ipv4(ip) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.version == 4 and a.is_global


def _unreachable(detail: str, port=None, ip=None, verified=None, mapping=None) -> Decision:
    if mapping is not None:
        try:
            mapping.release()
        except Exception:
            pass
    return Decision("unreachable", ip, port, verified, detail)


def forward_hint(local_port: int, dht_port: int | None = None) -> str:
    udp = f" and UDP {dht_port}" if dht_port else " and the DHT's UDP port"
    return (f"forward TCP {local_port}{udp} on your router to this machine, then run with "
            f"--public-ip YOUR.PUBLIC.IP --external-port {local_port}")


def decide(cfg: DirectConfig, local_port: int, dial_back, *, mapper=nat.map_port, stun=nat.stun_report,
           log=lambda m: None) -> Decision:
    """The address this node publishes, or an `unreachable` decision. `dial_back(ip, port) -> bool | None`."""
    if cfg.loopback:
        return Decision("loopback", LOOPBACK_IP, local_port, None, "test mode: loopback")

    def public_ip_from_stun():
        try:
            return stun().get("public_ip")
        except Exception as e:
            log(f"STUN failed ({type(e).__name__})")
            return None

    mapping = None
    if cfg.public_ip or cfg.external_port:
        kind = "manual"
        ip = cfg.public_ip or public_ip_from_stun()
        port = cfg.external_port or local_port
        if not ip:
            return _unreachable("could not find this machine's public IP (STUN got no answer); pass --public-ip", port)
    else:
        try:
            mapping = mapper(local_port)
            kind, ip, port = "mapped", mapping.external_ip, mapping.external_port
            log(f"the router mapped TCP {port} ({mapping.method})")
            if not is_public_ipv4(ip):   # a router behind another NAT reports a private address
                ip = public_ip_from_stun()
        except nat.NatError as e:
            kind, port = "stun", local_port
            log(f"no automatic port mapping ({str(e)[:120]}); trying the public address as is")
            ip = public_ip_from_stun()
        except Exception as e:
            kind, port = "stun", local_port
            log(f"port mapping failed ({type(e).__name__}); trying the public address as is")
            ip = public_ip_from_stun()
        if not ip:
            return _unreachable("could not find this machine's public IP (STUN got no answer)", port, mapping=mapping)
    if not is_public_ipv4(ip):
        return _unreachable(f"{ip} is not a public IPv4 address (private, loopback or carrier-grade NAT)", port, ip,
                            mapping=mapping)
    report = nat.check_reachability(ip, port, dial_back)
    if report.direct:
        return Decision(kind, ip, port, True, report.detail, mapping)
    asked = [a for a in report.answers if a is not None]
    if not asked and kind == "manual":
        # Nobody could be asked (a lone first node): trust the owner's explicit forward, loudly.
        return Decision(kind, ip, port, None, "no remote peer could check the forwarded port; published unverified",
                        mapping)
    return _unreachable(report.detail, port, ip, verified=False if asked else None, mapping=mapping)
