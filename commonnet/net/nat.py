"""Direct connectivity: what does this network allow, and open a port in the router if it lets us.

Four jobs, stdlib only, every network call injectable so tests need no network and none can hang
(each call has a 2-3 second timeout):

1. STUN (RFC 5389): learn the public IP and port, and whether the NAT keeps the same mapping for
   different destinations (nat_kind).
2. Port mapping: UPnP IGD (SSDP, device description, SOAP), then PCP (RFC 6887) and NAT-PMP
   (RFC 6886), whichever the router speaks. map_port() returns a Mapping that renews itself and is
   removed by release().
3. local_addresses() and default_gateway().
4. check_reachability(): ask other peers (a caller-supplied dial-back function) whether they can
   reach the mapped address, and decide direct or unreachable.

Run `python -m commonnet.net.nat` to see what this machine's network allows. It keeps no mapping.
Add --try-map to create one mapping on a high port, check it and delete it.

Public IP detection uses STUN only, never an HTTP "what is my IP" service. IPv4 only for now.
"""
from __future__ import annotations

import argparse
import dataclasses
import http.client
import ipaddress
import json
import os
import random
import re
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET

STUN_SERVERS = (("stun.l.google.com", 19302), ("stun1.l.google.com", 19302), ("stun.cloudflare.com", 3478))
STUN_TIMEOUT_SECONDS = 2.0
SSDP_TIMEOUT_SECONDS = 2.0
HTTP_TIMEOUT_SECONDS = 3.0
PMP_TIMEOUT_SECONDS = 1.0  # per try; RFC 6886 starts at 250 ms and doubles
PMP_TRIES = 2
RESOLVE_TIMEOUT_SECONDS = 2.0
MAX_HTTP_BYTES = 256 * 1024
DEFAULT_LEASE_SECONDS = 3600
EPHEMERAL_PORTS = (49152, 65535)

NAT_KINDS = ("open", "cone", "symmetric", "unknown")


class NatError(Exception):
    pass


class MappingFailed(NatError):
    """No method could create a mapping. `errors` maps method name to why."""

    def __init__(self, errors: dict):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()) or "no method available")
        self.errors = errors


# --------------------------------------------------------------------------- plumbing

def udp_exchange(addr, packet: bytes, timeout: float, *, tries: int = 1, sock=None) -> bytes | None:
    """Send one datagram and return the first reply from addr's IP, or None. Never longer than timeout
    in total. Pass `sock` to send from a chosen local socket (keeps the same source port across servers)."""
    own = sock is None
    s = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        deadline = time.monotonic() + timeout
        slice_ = timeout / max(1, tries)
        for i in range(max(1, tries)):
            s.sendto(packet, addr)
            until = min(deadline, time.monotonic() + slice_)
            while True:
                left = until - time.monotonic()
                if left <= 0:
                    break
                s.settimeout(left)
                try:
                    data, src = s.recvfrom(4096)
                except (socket.timeout, BlockingIOError):
                    break
                if src[0] == addr[0]:
                    return data
        return None
    except OSError:
        return None
    finally:
        if own:
            s.close()


def http_request(method: str, url: str, headers: dict | None = None, body: bytes | None = None,
                 timeout: float = HTTP_TIMEOUT_SECONDS) -> tuple:
    """(status, body bytes). Plain http only (a LAN router). Raises OSError / HTTPException on failure."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        raise NatError(f"not an http URL: {url!r}")
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        conn.request(method, target, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read(MAX_HTTP_BYTES)
    finally:
        conn.close()


def _resolve(host: str, timeout: float = RESOLVE_TIMEOUT_SECONDS, resolver=socket.getaddrinfo) -> str | None:
    """IPv4 address for host within timeout, or None. getaddrinfo cannot be interrupted, so it runs in a
    daemon thread that is abandoned on timeout."""
    try:
        socket.inet_aton(host)
        return host
    except OSError:
        pass
    box: list = []

    def work():
        try:
            box.append(resolver(host, None, family=socket.AF_INET, type=socket.SOCK_DGRAM)[0][4][0])
        except (OSError, IndexError):
            box.append(None)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    return box[0] if box else None


# --------------------------------------------------------------------------- local addresses

def local_addresses(*, probe=None) -> list:
    """This machine's non-loopback IPv4 addresses (LAN IPs), best guess first.

    The first is the one the OS would use for outbound traffic, found by connecting a UDP socket (no
    packet is sent). Others come from the host name."""
    found: list = []

    def add(ip):
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            return
        if a.version == 4 and not a.is_loopback and not a.is_link_local and not a.is_unspecified and ip not in found:
            found.append(ip)

    if probe is not None:
        for ip in probe():
            add(ip)
        return found
    for target in ("192.0.2.1", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((target, 9))
                add(s.getsockname()[0])
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    return found


def parse_route_get_default(text: str) -> str | None:
    """Gateway from macOS/BSD `route -n get default` output."""
    m = re.search(r"^\s*gateway:\s*(\d+\.\d+\.\d+\.\d+)\s*$", text, re.M)
    return m.group(1) if m else None


def parse_proc_net_route(text: str) -> str | None:
    """Gateway from Linux /proc/net/route (little-endian hex)."""
    for line in text.splitlines()[1:]:
        f = line.split()
        if len(f) >= 3 and f[1] == "00000000" and int(f[3], 16) & 2:
            return socket.inet_ntoa(struct.pack("<L", int(f[2], 16)))
    return None


def default_gateway(*, run=None, read=None, locals_=None) -> str | None:
    """The router's LAN address, for NAT-PMP/PCP. Falls back to x.y.z.1 of the first LAN address."""
    gw = None
    try:
        if sys.platform.startswith("linux"):
            gw = parse_proc_net_route((read or (lambda p: open(p).read()))("/proc/net/route"))
        elif sys.platform in ("darwin",) or "bsd" in sys.platform:
            out = (run or (lambda: subprocess.run(["route", "-n", "get", "default"], capture_output=True,
                                                  text=True, timeout=3).stdout))()
            gw = parse_route_get_default(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        gw = None
    if gw:
        return gw
    addrs = locals_ if locals_ is not None else local_addresses()
    for ip in addrs:
        if ipaddress.ip_address(ip).is_private:
            return ".".join(ip.split(".")[:3] + ["1"])
    return None


# --------------------------------------------------------------------------- STUN (RFC 5389)

STUN_MAGIC = 0x2112A442
STUN_BINDING_REQUEST = 0x0001
STUN_BINDING_SUCCESS = 0x0101
ATTR_MAPPED_ADDRESS = 0x0001
ATTR_XOR_MAPPED_ADDRESS = 0x0020


def stun_binding_request(txid: bytes | None = None) -> tuple:
    """(packet, txid): a Binding request with no attributes."""
    txid = txid or secrets.token_bytes(12)
    if len(txid) != 12:
        raise ValueError("a STUN transaction id is 12 bytes")
    return struct.pack(">HHI", STUN_BINDING_REQUEST, 0, STUN_MAGIC) + txid, txid


def parse_binding_response(data: bytes, txid: bytes) -> tuple | None:
    """(ip, port) from a Binding success response for this transaction, or None if it is anything else
    (wrong transaction, error, truncated, no address). Prefers XOR-MAPPED-ADDRESS."""
    if len(data) < 20:
        return None
    mtype, length, magic = struct.unpack(">HHI", data[:8])
    if mtype != STUN_BINDING_SUCCESS or magic != STUN_MAGIC or data[8:20] != txid or len(data) < 20 + length:
        return None
    body, pos, mapped, xor = data[20:20 + length], 0, None, None
    while pos + 4 <= len(body):
        atype, alen = struct.unpack(">HH", body[pos:pos + 4])
        value = body[pos + 4:pos + 4 + alen]
        if len(value) < alen:
            return None
        if atype in (ATTR_MAPPED_ADDRESS, ATTR_XOR_MAPPED_ADDRESS) and len(value) >= 4:
            family, port = value[1], struct.unpack(">H", value[2:4])[0]
            raw = value[4:]
            addr = None
            if family == 1 and len(raw) == 4:
                if atype == ATTR_XOR_MAPPED_ADDRESS:
                    port ^= STUN_MAGIC >> 16
                    raw = bytes(a ^ b for a, b in zip(raw, struct.pack(">I", STUN_MAGIC)))
                addr = (socket.inet_ntop(socket.AF_INET, raw), port)
            elif family == 2 and len(raw) == 16:
                if atype == ATTR_XOR_MAPPED_ADDRESS:
                    port ^= STUN_MAGIC >> 16
                    raw = bytes(a ^ b for a, b in zip(raw, struct.pack(">I", STUN_MAGIC) + txid))
                addr = (socket.inet_ntop(socket.AF_INET6, raw), port)
            if addr is not None:
                if atype == ATTR_XOR_MAPPED_ADDRESS:
                    xor = addr
                else:
                    mapped = addr
        pos += 4 + ((alen + 3) & ~3)
    return xor or mapped


def stun_query(server: tuple, *, sock=None, timeout: float = STUN_TIMEOUT_SECONDS, exchange=udp_exchange,
               resolve=_resolve) -> tuple | None:
    """(ip, port) the STUN server saw this socket as, or None (no answer, unresolvable, bad reply)."""
    ip = resolve(server[0])
    if ip is None:
        return None
    packet, txid = stun_binding_request()
    data = exchange((ip, server[1]), packet, timeout, tries=2, sock=sock)
    return parse_binding_response(data, txid) if data else None


def classify_nat(mappings: list, local_ips: list, local_port: int | None = None) -> str:
    """Decide the NAT's UDP behaviour from the (ip, port) different STUN servers reported for ONE local socket.

    open       the public address is one of this machine's own (no NAT in the way, at least for IPv4)
    cone       two or more servers saw the same (ip, port): the mapping does not depend on the destination,
               so hole punching and a port mapping can work. (Full, restricted and port-restricted cone are
               not told apart: that needs CHANGE-REQUEST, which most public servers no longer answer.)
    symmetric  servers saw different ports: each destination gets its own mapping, so a peer cannot
               predict the port it would need to be reached on
    unknown    fewer than two answers, so stability cannot be judged
    """
    seen = [m for m in mappings if m]
    if not seen:
        return "unknown"
    if any(m[0] in local_ips for m in seen):
        return "open"
    if len(seen) < 2:
        return "unknown"
    return "cone" if len(set(seen)) == 1 else "symmetric"


def stun_report(servers=STUN_SERVERS, *, local_ips=None, exchange=udp_exchange, resolve=_resolve,
                timeout: float = STUN_TIMEOUT_SECONDS) -> dict:
    """{public_ip, mapped_port, nat_kind, mappings}: queries each server from one UDP socket. Stops once two
    servers have answered (enough to judge) so the usual cost is one or two round trips."""
    local_ips = local_addresses() if local_ips is None else local_ips
    mappings: list = []
    sock = None
    try:
        if exchange is udp_exchange:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("", 0))
        for server in servers:
            m = stun_query(server, sock=sock, timeout=timeout, exchange=exchange, resolve=resolve)
            if m and ":" not in m[0]:
                mappings.append(m)
            if len(mappings) >= 2:
                break
    finally:
        if sock is not None:
            sock.close()
    kind = classify_nat(mappings, local_ips)
    first = mappings[0] if mappings else None
    return {"public_ip": first[0] if first else None, "mapped_port": first[1] if first else None,
            "nat_kind": kind, "mappings": [list(m) for m in mappings]}


# --------------------------------------------------------------------------- Mapping

@dataclasses.dataclass
class Mapping:
    """An open port in the router. Keep it while serving; release() deletes it and stops the renewal."""
    external_ip: str | None
    external_port: int
    internal_port: int
    method: str
    lease_seconds: int = 0
    _renew: object = dataclasses.field(default=None, repr=False)
    _delete: object = dataclasses.field(default=None, repr=False)
    _stop: threading.Event = dataclasses.field(default_factory=threading.Event, repr=False)
    released: bool = False
    last_error: str = ""

    def start_renewal(self) -> None:
        if not self._renew or self.lease_seconds <= 0:
            return
        interval = max(30.0, self.lease_seconds / 2)

        def loop():
            while not self._stop.wait(interval):
                try:
                    self._renew()
                    self.last_error = ""
                except Exception as e:  # keep trying; the lease may still be live
                    self.last_error = f"{type(e).__name__}: {e}"[:200]

        threading.Thread(target=loop, name="nat-renew", daemon=True).start()

    def release(self) -> bool:
        """Delete the mapping. Idempotent. True if the router confirmed the delete."""
        if self.released:
            return True
        self.released = True
        self._stop.set()
        try:
            if self._delete:
                self._delete()
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return False

    @property
    def endpoint(self) -> str | None:
        return f"{self.external_ip}:{self.external_port}" if self.external_ip else None


def _pick_ports(preferred: int | None) -> list:
    ports = [preferred] if preferred and 1024 <= preferred <= 65535 else []
    while len(ports) < 4:
        p = random.randint(*EPHEMERAL_PORTS)
        if p not in ports:
            ports.append(p)
    return ports


# --------------------------------------------------------------------------- UPnP IGD

SSDP_ADDR = ("239.255.255.250", 1900)
SEARCH_TARGETS = ("urn:schemas-upnp-org:service:WANIPConnection:1",
                  "urn:schemas-upnp-org:service:WANPPPConnection:1",
                  "urn:schemas-upnp-org:device:InternetGatewayDevice:1")
WAN_SERVICE_PREFIXES = ("urn:schemas-upnp-org:service:WANIPConnection:", "urn:schemas-upnp-org:service:WANPPPConnection:")


def ssdp_request(target: str) -> bytes:
    return ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\n"
            f"MX: 1\r\nST: {target}\r\n\r\n").encode("ascii")


def parse_ssdp_response(data: bytes) -> dict | None:
    """Header dict (lowercase names) from an SSDP reply, or None if it is not a 200 reply."""
    try:
        text = data.decode("utf-8", "replace")
    except Exception:
        return None
    lines = text.split("\r\n")
    if not lines or " 200" not in lines[0]:
        return None
    headers = {}
    for line in lines[1:]:
        k, sep, v = line.partition(":")
        if sep:
            headers[k.strip().lower()] = v.strip()
    return headers if "location" in headers else None


def ssdp_collect(timeout: float = SSDP_TIMEOUT_SECONDS) -> list:
    """Datagrams answering an M-SEARCH for an IGD within `timeout` (multicast, so there can be several)."""
    replies: list = []
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        for target in SEARCH_TARGETS:
            s.sendto(ssdp_request(target), SSDP_ADDR)
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            s.settimeout(left)
            try:
                replies.append(s.recvfrom(4096)[0])
            except (socket.timeout, BlockingIOError):
                break
    except OSError:
        pass
    finally:
        s.close()
    return replies


def _is_lan_url(url: str) -> bool:
    """A router's control URL must point at a private LAN address: a hostile SSDP reply must not make
    this code send requests to arbitrary hosts."""
    try:
        host = urllib.parse.urlsplit(url).hostname
        a = ipaddress.ip_address(host)
    except (ValueError, TypeError):
        return False
    return a.version == 4 and (a.is_private or a.is_link_local) and not a.is_loopback


def ssdp_locations(replies: list) -> list:
    out: list = []
    for data in replies:
        h = parse_ssdp_response(data)
        if h and h["location"] not in out and _is_lan_url(h["location"]):
            out.append(h["location"])
    return out


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_xml(data: bytes):
    """Parse XML from the LAN. Documents with a DTD are refused (entity expansion attacks)."""
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise NatError("XML with a DTD refused")
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise NatError(f"bad XML: {e}") from None


def parse_device_description(data: bytes, location: str) -> list:
    """[(service_type, absolute control_url)] for each WANIPConnection / WANPPPConnection service."""
    root = _safe_xml(data)
    base = location
    for el in root.iter():
        if _strip_ns(el.tag) == "URLBase" and el.text and el.text.strip():
            base = el.text.strip()
    found: list = []
    for el in root.iter():
        if _strip_ns(el.tag) != "service":
            continue
        fields = {_strip_ns(c.tag): (c.text or "").strip() for c in el}
        stype, ctrl = fields.get("serviceType", ""), fields.get("controlURL", "")
        if stype.startswith(WAN_SERVICE_PREFIXES) and ctrl:
            url = urllib.parse.urljoin(base, ctrl)
            if _is_lan_url(url):
                found.append((stype, url))
    return found


def soap_envelope(service_type: str, action: str, args: dict) -> bytes:
    from xml.sax.saxutils import escape
    inner = "".join(f"<{k}>{escape(str(v))}</{k}>" for k, v in args.items())
    return ('<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
            's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body>'
            f'<u:{action} xmlns:u="{service_type}">{inner}</u:{action}></s:Body></s:Envelope>').encode("utf-8")


def parse_soap_response(data: bytes) -> dict:
    """The fields of a SOAP reply. Raises NatError("UPnP error CODE: text") for a fault."""
    root = _safe_xml(data)
    fault = [el for el in root.iter() if _strip_ns(el.tag) == "UPnPError"]
    if fault:
        f = {_strip_ns(c.tag): (c.text or "").strip() for c in fault[0]}
        err = NatError(f"UPnP error {f.get('errorCode', '?')}: {f.get('errorDescription', '')}")
        err.code = int(f["errorCode"]) if f.get("errorCode", "").isdigit() else None
        raise err
    out: dict = {}
    for el in root.iter():
        if len(el) == 0 and _strip_ns(el.tag) not in ("Envelope", "Body"):
            out[_strip_ns(el.tag)] = (el.text or "").strip()
    return out


class UpnpClient:
    """One IGD service: GetExternalIPAddress, AddPortMapping, DeletePortMapping."""

    def __init__(self, service_type: str, control_url: str, *, http=http_request, timeout=HTTP_TIMEOUT_SECONDS):
        self.service_type, self.control_url, self._http, self._timeout = service_type, control_url, http, timeout

    def call(self, action: str, args: dict | None = None) -> dict:
        headers = {"Content-Type": 'text/xml; charset="utf-8"', "SOAPAction": f'"{self.service_type}#{action}"'}
        status, body = self._http("POST", self.control_url, headers, soap_envelope(self.service_type, action, args or {}),
                                  self._timeout)
        if status >= 400 and not body:
            raise NatError(f"HTTP {status}")
        return parse_soap_response(body)

    def external_ip(self) -> str:
        return self.call("GetExternalIPAddress").get("NewExternalIPAddress", "")

    def add(self, internal_ip: str, internal_port: int, external_port: int, lease: int, description: str) -> None:
        self.call("AddPortMapping", {
            "NewRemoteHost": "", "NewExternalPort": external_port, "NewProtocol": "TCP",
            "NewInternalPort": internal_port, "NewInternalClient": internal_ip, "NewEnabled": 1,
            "NewPortMappingDescription": description, "NewLeaseDuration": lease})

    def delete(self, external_port: int) -> None:
        self.call("DeletePortMapping", {"NewRemoteHost": "", "NewExternalPort": external_port, "NewProtocol": "TCP"})

    def has(self, external_port: int) -> bool:
        try:
            self.call("GetSpecificPortMappingEntry", {"NewRemoteHost": "", "NewExternalPort": external_port,
                                                      "NewProtocol": "TCP"})
            return True
        except NatError:
            return False


def upnp_discover(*, collect=ssdp_collect, http=http_request, timeout=HTTP_TIMEOUT_SECONDS) -> list:
    """UpnpClient for each IGD WAN service found on the LAN. Never raises; an unreachable device is skipped."""
    clients: list = []
    for location in ssdp_locations(collect()):
        try:
            status, body = http("GET", location, None, None, timeout)
            if status != 200:
                continue
            for stype, url in parse_device_description(body, location):
                clients.append(UpnpClient(stype, url, http=http, timeout=timeout))
        except (OSError, http_client_errors, NatError):
            continue
    return clients


http_client_errors = http.client.HTTPException


def upnp_map(local_port: int, local_ip: str, *, preferred_port: int | None = None, lease: int = DEFAULT_LEASE_SECONDS,
             description: str = "commonnet", clients: list | None = None, **discover_kw) -> Mapping:
    clients = clients if clients is not None else upnp_discover(**discover_kw)
    if not clients:
        raise NatError("no UPnP gateway answered")
    last: Exception = NatError("no attempt")
    for client in clients:
        for port in _pick_ports(preferred_port):
            used_lease = lease
            for _ in range(2):
                try:
                    client.add(local_ip, local_port, port, used_lease, description)
                except NatError as e:
                    last = e
                    if getattr(e, "code", None) == 725 and used_lease:  # permanent leases only
                        used_lease = 0
                        continue
                    if getattr(e, "code", None) in (718, 727):  # port taken: try another
                        break
                    break
                except (OSError, http_client_errors) as e:
                    last = e
                    break
                try:
                    ip = client.external_ip()
                except (NatError, OSError, http_client_errors):
                    ip = None
                m = Mapping(ip or None, port, local_port, "upnp", used_lease)
                m._renew = lambda c=client, p=port, l=used_lease: c.add(local_ip, local_port, p, l, description)
                m._delete = lambda c=client, p=port: c.delete(p)
                m.start_renewal()
                return m
    raise NatError(f"UPnP refused: {last}")


# --------------------------------------------------------------------------- NAT-PMP (RFC 6886) and PCP (RFC 6887)

PMP_PORT = 5351
PMP_RESULTS = {0: "success", 1: "unsupported version", 2: "not authorized or disabled", 3: "network failure",
               4: "out of resources", 5: "unsupported opcode"}


def pmp_external_request() -> bytes:
    return struct.pack(">BB", 0, 0)


def parse_pmp_external_response(data: bytes) -> str:
    if len(data) != 12:
        raise NatError("bad NAT-PMP reply length")
    ver, op, result, _epoch = struct.unpack(">BBHI", data[:8])
    if ver != 0 or op != 128:
        raise NatError("not a NAT-PMP external address reply")
    if result:
        raise NatError(f"NAT-PMP: {PMP_RESULTS.get(result, result)}")
    return socket.inet_ntoa(data[8:12])


def pmp_map_request(internal_port: int, external_port: int, lifetime: int) -> bytes:
    return struct.pack(">BBHHHI", 0, 2, 0, internal_port, external_port, lifetime)  # opcode 2 is TCP


def parse_pmp_map_response(data: bytes) -> tuple:
    """(internal_port, external_port, lifetime). Raises NatError on a refusal."""
    if len(data) != 16:
        raise NatError("bad NAT-PMP reply length")
    ver, op, result, _epoch, internal, external, lifetime = struct.unpack(">BBHIHHI", data)
    if ver != 0 or op != 130:
        raise NatError("not a NAT-PMP TCP map reply")
    if result:
        raise NatError(f"NAT-PMP: {PMP_RESULTS.get(result, result)}")
    return internal, external, lifetime


def pmp_map(local_port: int, gateway: str, *, preferred_port: int | None = None, lease: int = DEFAULT_LEASE_SECONDS,
            exchange=udp_exchange) -> Mapping:
    gw = (gateway, PMP_PORT)

    def ask(packet):
        data = exchange(gw, packet, PMP_TIMEOUT_SECONDS * PMP_TRIES, tries=PMP_TRIES)
        if data is None:
            raise NatError("gateway did not answer NAT-PMP")
        return data

    ip = parse_pmp_external_response(ask(pmp_external_request()))
    last: Exception = NatError("no attempt")
    for port in _pick_ports(preferred_port)[:2]:
        try:
            _, ext, life = parse_pmp_map_response(ask(pmp_map_request(local_port, port, lease)))
        except NatError as e:
            last = e
            continue
        m = Mapping(ip, ext, local_port, "nat-pmp", life)
        m._renew = lambda e=ext: parse_pmp_map_response(ask(pmp_map_request(local_port, e, lease)))
        m._delete = lambda: parse_pmp_map_response(ask(pmp_map_request(local_port, 0, 0)))
        m.start_renewal()
        return m
    raise last


PCP_RESULTS = {0: "success", 1: "unsupported version", 2: "not authorized", 3: "malformed request",
               4: "unsupported opcode", 5: "unsupported option", 6: "malformed option", 7: "network failure",
               8: "no resources", 9: "unsupported protocol", 10: "user quota exceeded",
               11: "cannot provide suggested port", 12: "address mismatch", 13: "excessive remote peers"}


def _v4_mapped(ip: str) -> bytes:
    return b"\x00" * 10 + b"\xff\xff" + socket.inet_aton(ip)


def pcp_map_request(client_ip: str, internal_port: int, external_port: int, lifetime: int, nonce: bytes) -> bytes:
    """A 60-byte PCP MAP request for TCP."""
    header = struct.pack(">BBHI", 2, 1, 0, lifetime) + _v4_mapped(client_ip)
    op = nonce + struct.pack(">B3xHH", 6, internal_port, external_port) + b"\x00" * 16
    return header + op


def parse_pcp_map_response(data: bytes, nonce: bytes) -> tuple:
    """(external_ip, external_port, lifetime). Raises NatError on refusal or a reply for another request."""
    if len(data) < 60:
        raise NatError("short PCP reply")
    ver, rop, _res, result, lifetime = struct.unpack(">BBBBI", data[:8])
    if ver != 2 or not rop & 0x80 or rop & 0x7F != 1:
        raise NatError("not a PCP MAP reply")
    if result:
        raise NatError(f"PCP: {PCP_RESULTS.get(result, result)}")
    if data[24:36] != nonce:
        raise NatError("PCP reply does not match the request")
    ext_port = struct.unpack(">H", data[42:44])[0]
    raw = data[44:60]
    ip = socket.inet_ntoa(raw[12:]) if raw[:12] == b"\x00" * 10 + b"\xff\xff" else socket.inet_ntop(socket.AF_INET6, raw)
    return ip, ext_port, lifetime


def pcp_map(local_port: int, local_ip: str, gateway: str, *, preferred_port: int | None = None,
            lease: int = DEFAULT_LEASE_SECONDS, exchange=udp_exchange) -> Mapping:
    nonce = secrets.token_bytes(12)
    port = _pick_ports(preferred_port)[0]
    data = exchange((gateway, PMP_PORT), pcp_map_request(local_ip, local_port, port, lease, nonce),
                    PMP_TIMEOUT_SECONDS * PMP_TRIES, tries=PMP_TRIES)
    if data is None:
        raise NatError("gateway did not answer PCP")
    ip, ext, life = parse_pcp_map_response(data, nonce)

    def send(lifetime, ext_port):
        d = exchange((gateway, PMP_PORT), pcp_map_request(local_ip, local_port, ext_port, lifetime, nonce),
                     PMP_TIMEOUT_SECONDS * PMP_TRIES, tries=PMP_TRIES)
        if d is None:
            raise NatError("gateway did not answer PCP")
        return parse_pcp_map_response(d, nonce)

    m = Mapping(ip, ext, local_port, "pcp", life)
    m._renew = lambda: send(lease, ext)
    m._delete = lambda: send(0, ext)
    m.start_renewal()
    return m


# --------------------------------------------------------------------------- map_port

def map_port(local_port: int, *, methods=("upnp", "pcp", "nat-pmp"), preferred_port: int | None = None,
             lease: int = DEFAULT_LEASE_SECONDS, local_ip: str | None = None, gateway: str | None = None,
             upnp=upnp_map, pcp=pcp_map, pmp=pmp_map) -> Mapping:
    """Open TCP `local_port` in the router. Tries UPnP, then PCP, then NAT-PMP. Returns the first Mapping that
    works, already renewing itself in the background. Raises MappingFailed with each method's reason.

    The external address in the Mapping is the router's own idea of it; it may be a private address behind
    carrier-grade NAT. Use check_reachability() before believing it."""
    errors: dict = {}
    local_ip = local_ip or (local_addresses() or [None])[0]
    if local_ip is None:
        raise MappingFailed({"all": "no LAN address"})
    gateway = gateway or default_gateway()
    for method in methods:
        try:
            if method == "upnp":
                return upnp(local_port, local_ip, preferred_port=preferred_port, lease=lease)
            if gateway is None:
                errors[method] = "no gateway found"
            elif method == "pcp":
                return pcp(local_port, local_ip, gateway, preferred_port=preferred_port, lease=lease)
            elif method == "nat-pmp":
                return pmp(local_port, gateway, preferred_port=preferred_port, lease=lease)
        except (NatError, OSError, http_client_errors) as e:
            errors[method] = str(e) or type(e).__name__
    raise MappingFailed(errors)


# --------------------------------------------------------------------------- reachability

@dataclasses.dataclass
class ReachabilityReport:
    verdict: str                     # "direct" or "unreachable"
    address: tuple                   # (ip, port) that was tested
    answers: list                    # one entry per dial-back: True, False, or None (the helper could not be asked)
    detail: str = ""

    @property
    def direct(self) -> bool:
        return self.verdict == "direct"


def check_reachability(ip: str, port: int, dial_back, *, tries: int = 3) -> ReachabilityReport:
    """Ask `dial_back(ip, port) -> bool | None` up to `tries` times whether a remote peer can connect to
    ip:port. The integration step supplies a function that asks DHT peers (each call should pick a different
    peer). One True is enough for "direct". Otherwise "unreachable". None, or an exception, means the helper
    could not be asked; those are recorded but do not count as a refusal.

    A caller must make sure the listener is up first; one that answers TLS with the right pin is the
    strongest proof (see tcp_dial)."""
    answers: list = []
    for _ in range(max(1, tries)):
        try:
            a = dial_back(ip, port)
        except Exception:
            a = None
        answers.append(None if a is None else bool(a))
        if a:
            return ReachabilityReport("direct", (ip, port), answers, "a remote peer connected")
    asked = [a for a in answers if a is not None]
    detail = ("every remote peer failed to connect (blocked inbound, CGNAT, or no port mapping)" if asked
              else "no remote peer could be asked")
    return ReachabilityReport("unreachable", (ip, port), answers, detail)


def tcp_dial(ip: str, port: int, *, timeout: float = 3.0, requester_ip: str | None = None) -> bool:
    """What a helper peer runs when asked to test someone: can I open a TCP connection to ip:port?

    Abuse guard: if requester_ip (the source address the request really came from) is given, only that
    address may be tested, so the helper cannot be used to scan third parties. Private, loopback and
    link-local targets are refused always (a helper must not probe its own LAN)."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if not a.is_global or (requester_ip is not None and ip != requester_ip) or not 1 <= int(port) <= 65535:
        return False
    try:
        with socket.create_connection((ip, int(port)), timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- command line

def full_report(*, try_map: bool = False, stun=stun_report, local=local_addresses, gateway=default_gateway,
                discover=upnp_discover, pmp_probe=None, mapper=map_port) -> dict:
    """Everything this network allows, as a dict. Keeps no mapping."""
    out: dict = {"local_addresses": local()}
    out["stun"] = stun(local_ips=out["local_addresses"])
    gw = gateway()
    out["gateway"] = gw
    clients = discover()
    upnp: dict = {"available": bool(clients), "services": [c.service_type.rsplit(":", 2)[-2] for c in clients]}
    for c in clients[:1]:
        try:
            upnp["external_ip"] = c.external_ip()
        except (NatError, OSError, http_client_errors) as e:
            upnp["external_ip_error"] = str(e)[:100]
    out["upnp"] = upnp
    probe = pmp_probe or (lambda g: udp_exchange((g, PMP_PORT), pmp_external_request(), PMP_TIMEOUT_SECONDS * PMP_TRIES,
                                                 tries=PMP_TRIES))
    pmp: dict = {"available": False}
    if gw:
        data = probe(gw)
        if data:
            try:
                pmp = {"available": True, "external_ip": parse_pmp_external_response(data)}
            except NatError as e:
                pmp = {"available": False, "error": str(e)}
    out["nat_pmp"] = pmp
    if try_map:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # a real local listener so the mapping is meaningful
        sock.bind(("", 0))
        sock.listen(1)
        try:
            m = mapper(sock.getsockname()[1], preferred_port=random.randint(*EPHEMERAL_PORTS), lease=120,
                       local_ip=(out["local_addresses"] or [None])[0], gateway=gw)
            ok = m.release()
            out["mapping"] = {"created": True, "method": m.method, "external_ip": m.external_ip,
                              "external_port": m.external_port, "deleted": ok}
        except MappingFailed as e:
            out["mapping"] = {"created": False, "errors": e.errors}
        finally:
            sock.close()
    return out


def redact_ip(ip: str | None) -> str | None:
    return None if not ip else ip.split(".")[0] + ".x.x.x" if "." in ip else "redacted"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m commonnet.net.nat",
                                description="Show what this machine's network allows for direct connections. Keeps no mapping.")
    p.add_argument("--try-map", action="store_true",
                   help="create ONE port mapping on a random high port, check it, delete it")
    p.add_argument("--redact", action="store_true", help="show only the first octet of public IPs")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    report = full_report(try_map=args.try_map)
    if args.redact:
        report["stun"]["public_ip"] = redact_ip(report["stun"]["public_ip"])
        report["stun"]["mappings"] = [[redact_ip(m[0]), m[1]] for m in report["stun"]["mappings"]]
        for k in ("upnp", "nat_pmp"):
            if report[k].get("external_ip"):
                report[k]["external_ip"] = redact_ip(report[k]["external_ip"])
        if report.get("mapping", {}).get("external_ip"):
            report["mapping"]["external_ip"] = redact_ip(report["mapping"]["external_ip"])
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    s = report["stun"]
    print(f"local addresses : {', '.join(report['local_addresses']) or 'none'}")
    print(f"gateway         : {report['gateway']}")
    print(f"public IP (STUN): {s['public_ip']}   NAT kind: {s['nat_kind']}   mapped port: {s['mapped_port']}")
    u, n = report["upnp"], report["nat_pmp"]
    print(f"UPnP IGD        : {'available ' + str(u.get('services')) if u['available'] else 'not found'}"
          + (f", router says external IP {u['external_ip']}" if u.get("external_ip") else ""))
    print(f"NAT-PMP/PCP     : {'available' if n['available'] else 'no answer'}"
          + (f" ({n['error']})" if n.get("error") else ""))
    m = report.get("mapping")
    if m is not None:
        print("mapping test    : " + (f"created via {m['method']} then deleted={m['deleted']}" if m["created"]
                                      else f"could not create ({m['errors']})"))
    else:
        print("mapping test    : not run (add --try-map)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
