"""Hostname resolution with a fallback to public DNS.

Some resolvers (Tailscale DNS on the coordinator Mac) answer NXDOMAIN for a brand-new
*.trycloudflare.com name for minutes. resolve() asks the system resolver first and only if
that fails (gaierror) asks 1.1.1.1, then 8.8.8.8, directly over UDP. Stdlib only.
Results from the fallback are cached briefly. A system success returns the hostname
unchanged, so normal setups are untouched.
"""
from __future__ import annotations

import random
import socket
import struct
import time

FALLBACK_SERVERS = (("1.1.1.1", 53), ("8.8.8.8", 53))
QUERY_TIMEOUT_SECONDS = 3.0
CACHE_SECONDS = 60.0

_cache: dict = {}


def build_query(hostname: str, query_id: int) -> bytes:
    header = struct.pack(">HHHHHH", query_id, 0x0100, 1, 0, 0, 0)
    name = b"".join(bytes([len(p)]) + p.encode("ascii") for p in hostname.rstrip(".").split("."))
    return header + name + b"\x00" + struct.pack(">HH", 1, 1)


def _skip_name(data: bytes, pos: int) -> int:
    while True:
        length = data[pos]
        if length == 0:
            return pos + 1
        if length & 0xC0 == 0xC0:
            return pos + 2
        pos += 1 + length


def parse_answer(data: bytes, query_id: int) -> str | None:
    """First A record address in the response, or None."""
    if len(data) < 12:
        return None
    rid, flags, qd, an = struct.unpack(">HHHH", data[:8])
    if rid != query_id or flags & 0x000F != 0:
        return None
    pos = 12
    for _ in range(qd):
        pos = _skip_name(data, pos) + 4
    for _ in range(an):
        pos = _skip_name(data, pos)
        rtype, _cls, _ttl, rdlen = struct.unpack(">HHIH", data[pos:pos + 10])
        pos += 10
        if rtype == 1 and rdlen == 4:
            return socket.inet_ntoa(data[pos:pos + 4])
        pos += rdlen
    return None


def query_a(hostname: str, server: tuple, timeout: float = QUERY_TIMEOUT_SECONDS) -> str | None:
    query_id = random.randrange(0x10000)
    packet = build_query(hostname, query_id)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(packet, server)
        try:
            data, _ = sock.recvfrom(2048)
        except OSError:
            return None
    try:
        return parse_answer(data, query_id)
    except (struct.error, IndexError):
        return None


def _is_ip(host: str) -> bool:
    try:
        socket.inet_aton(host)
        return host.replace(".", "").isdigit()
    except OSError:
        return ":" in host


def resolve(hostname: str, port: int = 443, *, system=None, servers=None, now=time.monotonic) -> str:
    """The host to connect to: hostname itself if the system resolver knows it, else an IPv4 string.

    Raises socket.gaierror if neither the system nor the fallback servers can resolve it.
    """
    if _is_ip(hostname):
        return hostname
    system = system or socket.getaddrinfo
    try:
        system(hostname, port, type=socket.SOCK_STREAM)
        return hostname
    except socket.gaierror as error:
        failure = error
    hit = _cache.get(hostname)
    if hit and now() < hit[1]:
        return hit[0]
    for server in (servers if servers is not None else FALLBACK_SERVERS):
        try:
            ip = query_a(hostname, server)
        except (OSError, UnicodeError):
            ip = None
        if ip:
            _cache[hostname] = (ip, now() + CACHE_SECONDS)
            return ip
    raise failure


def clear_cache() -> None:
    _cache.clear()
