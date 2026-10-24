"""NAT traversal helpers: every network call is faked, so these need no network."""
import socket
import struct
import threading
import time
import unittest

from commonnet.net import nat

MAGIC = struct.pack(">I", nat.STUN_MAGIC)
TXID = bytes(range(12))


def xor_attr(ip: str, port: int) -> bytes:
    xport = port ^ (nat.STUN_MAGIC >> 16)
    xip = bytes(a ^ b for a, b in zip(socket.inet_aton(ip), MAGIC))
    value = struct.pack(">BBH", 0, 1, xport) + xip
    return struct.pack(">HH", nat.ATTR_XOR_MAPPED_ADDRESS, len(value)) + value


def plain_attr(ip: str, port: int) -> bytes:
    value = struct.pack(">BBH", 0, 1, port) + socket.inet_aton(ip)
    return struct.pack(">HH", nat.ATTR_MAPPED_ADDRESS, len(value)) + value


def stun_reply(attrs: bytes, txid=TXID, mtype=nat.STUN_BINDING_SUCCESS) -> bytes:
    return struct.pack(">HHI", mtype, len(attrs), nat.STUN_MAGIC) + txid + attrs


class StunParsing(unittest.TestCase):
    def test_request_shape(self):
        packet, txid = nat.stun_binding_request()
        self.assertEqual(len(packet), 20)
        self.assertEqual(packet[:2], b"\x00\x01")
        self.assertEqual(packet[4:8], MAGIC)
        self.assertEqual(packet[8:], txid)

    def test_rfc5769_style_xor_mapped(self):
        # 192.0.2.1:32853 is the RFC 5769 example address
        self.assertEqual(nat.parse_binding_response(stun_reply(xor_attr("192.0.2.1", 32853)), TXID),
                         ("192.0.2.1", 32853))

    def test_plain_mapped_and_xor_preferred(self):
        self.assertEqual(nat.parse_binding_response(stun_reply(plain_attr("203.0.113.9", 4000)), TXID),
                         ("203.0.113.9", 4000))
        both = plain_attr("10.0.0.1", 1) + xor_attr("203.0.113.9", 4000)
        self.assertEqual(nat.parse_binding_response(stun_reply(both), TXID), ("203.0.113.9", 4000))

    def test_ignores_unknown_attrs_with_padding(self):
        odd = struct.pack(">HH", 0x8022, 5) + b"hello\x00\x00\x00"  # SOFTWARE, padded to 4
        self.assertEqual(nat.parse_binding_response(stun_reply(odd + xor_attr("198.51.100.7", 99)), TXID),
                         ("198.51.100.7", 99))

    def test_ipv6_xor(self):
        raw = socket.inet_pton(socket.AF_INET6, "2001:db8::1")
        x = bytes(a ^ b for a, b in zip(raw, MAGIC + TXID))
        value = struct.pack(">BBH", 0, 2, 5000 ^ (nat.STUN_MAGIC >> 16)) + x
        attr = struct.pack(">HH", nat.ATTR_XOR_MAPPED_ADDRESS, len(value)) + value
        self.assertEqual(nat.parse_binding_response(stun_reply(attr), TXID), ("2001:db8::1", 5000))

    def test_rejects_bad_packets(self):
        good = stun_reply(xor_attr("192.0.2.1", 1))
        for bad in (b"", good[:19], good[:-2], stun_reply(xor_attr("192.0.2.1", 1), txid=b"x" * 12),
                    stun_reply(xor_attr("192.0.2.1", 1), mtype=0x0111), stun_reply(b""),
                    good[:4] + b"\x00\x00\x00\x00" + good[8:],
                    stun_reply(struct.pack(">HH", nat.ATTR_MAPPED_ADDRESS, 40) + b"\x00" * 3)):
            self.assertIsNone(nat.parse_binding_response(bad, TXID), bad)

    def test_query_uses_exchange_and_survives_failures(self):
        def good(addr, packet, timeout, tries=1, sock=None):
            txid = packet[8:]
            return stun_reply(xor_attr("203.0.113.5", 1234), txid=txid)

        self.assertEqual(nat.stun_query(("h", 1), exchange=good, resolve=lambda h: "1.2.3.4"), ("203.0.113.5", 1234))
        self.assertIsNone(nat.stun_query(("h", 1), exchange=lambda *a, **k: None, resolve=lambda h: "1.2.3.4"))
        self.assertIsNone(nat.stun_query(("h", 1), exchange=good, resolve=lambda h: None))
        self.assertIsNone(nat.stun_query(("h", 1), exchange=lambda *a, **k: b"junk", resolve=lambda h: "1.2.3.4"))


class NatKind(unittest.TestCase):
    LOCAL = ["192.168.1.20"]

    def test_decisions(self):
        k = nat.classify_nat
        self.assertEqual(k([], self.LOCAL), "unknown")
        self.assertEqual(k([("203.0.113.1", 5)], self.LOCAL), "unknown")
        self.assertEqual(k([("203.0.113.1", 5), ("203.0.113.1", 5)], self.LOCAL), "cone")
        self.assertEqual(k([("203.0.113.1", 5), ("203.0.113.1", 6)], self.LOCAL), "symmetric")
        self.assertEqual(k([("203.0.113.1", 5), ("203.0.113.2", 5)], self.LOCAL), "symmetric")
        self.assertEqual(k([("203.0.113.1", 5)], ["203.0.113.1"]), "open")
        self.assertEqual(k([None, ("203.0.113.1", 5), ("203.0.113.1", 5)], self.LOCAL), "cone")

    def test_stun_report_two_servers(self):
        answers = {"a": ("203.0.113.1", 4000), "b": ("203.0.113.1", 4000), "c": ("203.0.113.1", 9)}
        calls = []

        def exchange(addr, packet, timeout, tries=1, sock=None):
            calls.append(addr[0])
            return stun_reply(xor_attr(*answers[addr[0]]), txid=packet[8:])

        r = nat.stun_report((("a", 1), ("b", 1), ("c", 1)), local_ips=self.LOCAL, exchange=exchange,
                            resolve=lambda h: h)
        self.assertEqual((r["public_ip"], r["mapped_port"], r["nat_kind"]), ("203.0.113.1", 4000, "cone"))
        self.assertEqual(calls, ["a", "b"])  # stops once two have answered

    def test_stun_report_symmetric_and_dead(self):
        ports = iter((1000, 2000))
        sym = lambda addr, packet, timeout, tries=1, sock=None: stun_reply(
            xor_attr("203.0.113.1", next(ports)), txid=packet[8:])
        r = nat.stun_report((("a", 1), ("b", 1)), local_ips=self.LOCAL, exchange=sym, resolve=lambda h: h)
        self.assertEqual(r["nat_kind"], "symmetric")
        dead = nat.stun_report((("a", 1),), local_ips=self.LOCAL, exchange=lambda *a, **k: None, resolve=lambda h: h)
        self.assertEqual((dead["public_ip"], dead["nat_kind"]), (None, "unknown"))


class UpnpXml(unittest.TestCase):
    SSDP = (b"HTTP/1.1 200 OK\r\nCACHE-CONTROL: max-age=120\r\nLOCATION: http://192.168.1.1:5000/rootDesc.xml\r\n"
            b"ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n")
    DESC = b"""<?xml version="1.0"?>
<root xmlns="urn:schemas-upnp-org:device-1-0"><device><deviceList><device><deviceList><device>
<serviceList>
 <service><serviceType>urn:schemas-upnp-org:service:WANCommonInterfaceConfig:1</serviceType><controlURL>/ctl/cic</controlURL></service>
 <service><serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType><controlURL>/ctl/IPConn</controlURL></service>
</serviceList></device></deviceList></device></deviceList></device></root>"""

    def test_ssdp(self):
        self.assertEqual(nat.parse_ssdp_response(self.SSDP)["location"], "http://192.168.1.1:5000/rootDesc.xml")
        self.assertIsNone(nat.parse_ssdp_response(b"M-SEARCH * HTTP/1.1\r\n\r\n"))
        self.assertIsNone(nat.parse_ssdp_response(b"HTTP/1.1 200 OK\r\n\r\n"))

    def test_locations_must_be_lan(self):
        evil = self.SSDP.replace(b"192.168.1.1", b"8.8.8.8")
        loop = self.SSDP.replace(b"192.168.1.1", b"127.0.0.1")
        self.assertEqual(nat.ssdp_locations([self.SSDP, self.SSDP, evil, loop]), ["http://192.168.1.1:5000/rootDesc.xml"])

    def test_description(self):
        self.assertEqual(nat.parse_device_description(self.DESC, "http://192.168.1.1:5000/rootDesc.xml"),
                         [("urn:schemas-upnp-org:service:WANIPConnection:1", "http://192.168.1.1:5000/ctl/IPConn")])

    def test_description_urlbase_and_hostile_control(self):
        xml = self.DESC.replace(b"<device>", b"<URLBase>http://192.168.1.1:49000/</URLBase><device>", 1)
        self.assertEqual(nat.parse_device_description(xml, "http://192.168.1.1:5000/x")[0][1],
                         "http://192.168.1.1:49000/ctl/IPConn")
        hostile = self.DESC.replace(b"/ctl/IPConn", b"http://8.8.8.8/ctl")
        self.assertEqual(nat.parse_device_description(hostile, "http://192.168.1.1:5000/x"), [])

    def test_dtd_and_garbage_refused(self):
        bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><root>&a;</root>'
        with self.assertRaises(nat.NatError):
            nat.parse_device_description(bomb, "http://192.168.1.1/")
        with self.assertRaises(nat.NatError):
            nat.parse_soap_response(b"<not xml")

    def test_soap_parse(self):
        ok = (b'<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"><s:Body>'
              b'<u:GetExternalIPAddressResponse xmlns:u="x"><NewExternalIPAddress>203.0.113.77</NewExternalIPAddress>'
              b'</u:GetExternalIPAddressResponse></s:Body></s:Envelope>')
        self.assertEqual(nat.parse_soap_response(ok)["NewExternalIPAddress"], "203.0.113.77")
        fault = (b'<s:Envelope xmlns:s="x"><s:Body><s:Fault><detail><UPnPError xmlns="urn:schemas-upnp-org:control-1-0">'
                 b'<errorCode>718</errorCode><errorDescription>ConflictInMappingEntry</errorDescription></UPnPError>'
                 b'</detail></s:Fault></s:Body></s:Envelope>')
        with self.assertRaises(nat.NatError) as cm:
            nat.parse_soap_response(fault)
        self.assertEqual(cm.exception.code, 718)

    def test_envelope_escapes(self):
        body = nat.soap_envelope("urn:x:1", "AddPortMapping", {"NewPortMappingDescription": "a<b>&"})
        self.assertIn(b"a&lt;b&gt;&amp;", body)
        self.assertIn(b'<u:AddPortMapping xmlns:u="urn:x:1">', body)


class FakeRouter:
    """A fake IGD: records SOAP actions; can be told to answer faults."""

    def __init__(self, faults=None):
        self.faults, self.actions, self.mappings = dict(faults or {}), [], {}

    def __call__(self, method, url, headers, body, timeout):
        if method == "GET":
            return 200, UpnpXml.DESC
        action = headers["SOAPAction"].strip('"').split("#")[1]
        text = body.decode()
        self.actions.append(action)
        port = int(text.split("<NewExternalPort>")[1].split("<")[0]) if "<NewExternalPort>" in text else None
        lease = int(text.split("<NewLeaseDuration>")[1].split("<")[0]) if "<NewLeaseDuration>" in text else None
        queue = self.faults.get(action)
        if queue:
            code = queue.pop(0) if isinstance(queue, list) else queue
            if code == 725 and lease == 0:
                code = None
            if code:
                return 500, (f'<s:Envelope xmlns:s="x"><s:Body><s:Fault><detail><UPnPError><errorCode>{code}</errorCode>'
                             f'<errorDescription>no</errorDescription></UPnPError></detail></s:Fault></s:Body></s:Envelope>').encode()
        if action == "AddPortMapping":
            self.mappings[port] = lease
        if action == "DeletePortMapping":
            self.mappings.pop(port, None)
        reply = "<NewExternalIPAddress>203.0.113.50</NewExternalIPAddress>" if action == "GetExternalIPAddress" else ""
        return 200, f'<s:Envelope xmlns:s="x"><s:Body><u:{action}Response xmlns:u="y">{reply}</u:{action}Response></s:Body></s:Envelope>'.encode()


class UpnpFlow(unittest.TestCase):
    def run_map(self, router, **kw):
        return nat.upnp_map(8080, "192.168.1.20", collect=lambda: [UpnpXml.SSDP], http=router, **kw)

    def test_map_and_release(self):
        r = FakeRouter()
        m = self.run_map(r, preferred_port=50000)
        self.assertEqual((m.method, m.external_ip, m.external_port), ("upnp", "203.0.113.50", 50000))
        self.assertEqual(r.mappings, {50000: 3600})
        self.assertTrue(m.release())
        self.assertEqual(r.mappings, {})
        self.assertTrue(m.release())  # idempotent
        self.assertEqual(r.actions.count("DeletePortMapping"), 1)

    def test_permanent_lease_only_router(self):
        r = FakeRouter({"AddPortMapping": [725]})
        m = self.run_map(r, preferred_port=50001)
        self.assertEqual(r.mappings[50001], 0)
        m.release()

    def test_conflict_tries_another_port(self):
        r = FakeRouter({"AddPortMapping": [718]})
        m = self.run_map(r, preferred_port=50002)
        self.assertNotEqual(m.external_port, 50002)
        m.release()

    def test_refusal_and_no_gateway(self):
        with self.assertRaises(nat.NatError):
            self.run_map(FakeRouter({"AddPortMapping": 501}))
        with self.assertRaises(nat.NatError):
            nat.upnp_map(1, "192.168.1.2", collect=lambda: [])

    def test_renewal_reissues_add(self):
        r = FakeRouter()
        m = self.run_map(r, preferred_port=50003)
        m._renew()
        self.assertEqual(r.actions.count("AddPortMapping"), 2)
        m.release()

    def test_discovery_skips_dead_devices(self):
        def boom(*a):
            raise OSError("down")
        self.assertEqual(nat.upnp_discover(collect=lambda: [UpnpXml.SSDP], http=boom), [])


class PmpPcp(unittest.TestCase):
    def test_pmp_packets(self):
        self.assertEqual(nat.pmp_external_request(), b"\x00\x00")
        self.assertEqual(nat.pmp_map_request(8080, 50000, 3600),
                         bytes([0, 2, 0, 0]) + struct.pack(">HHI", 8080, 50000, 3600))
        reply = struct.pack(">BBHI4s", 0, 128, 0, 77, socket.inet_aton("203.0.113.9"))
        self.assertEqual(nat.parse_pmp_external_response(reply), "203.0.113.9")
        m = struct.pack(">BBHIHHI", 0, 130, 0, 77, 8080, 50000, 3600)
        self.assertEqual(nat.parse_pmp_map_response(m), (8080, 50000, 3600))
        for bad in (struct.pack(">BBHIHHI", 0, 130, 2, 77, 8080, 0, 0), m[:-1],
                    struct.pack(">BBHIHHI", 0, 128, 0, 77, 8080, 1, 1)):
            with self.assertRaises(nat.NatError):
                nat.parse_pmp_map_response(bad)
        with self.assertRaises(nat.NatError):
            nat.parse_pmp_external_response(struct.pack(">BBHI4s", 0, 128, 3, 0, b"\0" * 4))

    class Gateway:
        def __init__(self):
            self.log = []

        def __call__(self, addr, packet, timeout, tries=1, sock=None):
            self.log.append(packet)
            if len(packet) == 2:
                return struct.pack(">BBHI4s", 0, 128, 0, 1, socket.inet_aton("203.0.113.9"))
            _, _, _, internal, ext, life = struct.unpack(">BBHHHI", packet)
            return struct.pack(">BBHIHHI", 0, 130, 0, 1, internal, ext or 50123, life)

    def test_pmp_map_release(self):
        g = self.Gateway()
        m = nat.pmp_map(8080, "192.168.1.1", preferred_port=50100, exchange=g)
        self.assertEqual((m.method, m.external_ip, m.external_port), ("nat-pmp", "203.0.113.9", 50100))
        m.release()
        self.assertEqual(struct.unpack(">BBHHHI", g.log[-1])[4:], (0, 0))  # lifetime 0 deletes

    def test_pmp_no_answer(self):
        with self.assertRaises(nat.NatError):
            nat.pmp_map(1, "192.168.1.1", exchange=lambda *a, **k: None)

    def test_pcp_packet_and_reply(self):
        nonce = bytes(range(12))
        req = nat.pcp_map_request("192.168.1.20", 8080, 50000, 3600, nonce)
        self.assertEqual(len(req), 60)
        self.assertEqual(req[:4], bytes([2, 1, 0, 0]))
        self.assertEqual(req[24:36], nonce)
        self.assertEqual(req[36], 6)
        self.assertEqual(struct.unpack(">HH", req[40:44]), (8080, 50000))
        reply = (struct.pack(">BBBBI", 2, 0x81, 0, 0, 3600) + b"\0" * 16 + nonce + bytes([6, 0, 0, 0])
                 + struct.pack(">HH", 8080, 50000) + nat._v4_mapped("203.0.113.9"))
        self.assertEqual(nat.parse_pcp_map_response(reply, nonce), ("203.0.113.9", 50000, 3600))
        with self.assertRaises(nat.NatError):
            nat.parse_pcp_map_response(reply, b"x" * 12)
        refused = reply[:3] + b"\x02" + reply[4:]
        with self.assertRaises(nat.NatError):
            nat.parse_pcp_map_response(refused, nonce)

    def test_pcp_map(self):
        def gw(addr, packet, timeout, tries=1, sock=None):
            nonce, port = packet[24:36], struct.unpack(">H", packet[42:44])[0]
            life = struct.unpack(">I", packet[4:8])[0]
            return (struct.pack(">BBBBI", 2, 0x81, 0, 0, life) + b"\0" * 16 + nonce + bytes([6, 0, 0, 0])
                    + struct.pack(">HH", 8080, port) + nat._v4_mapped("203.0.113.9"))
        m = nat.pcp_map(8080, "192.168.1.20", "192.168.1.1", preferred_port=50200, exchange=gw)
        self.assertEqual((m.method, m.external_port), ("pcp", 50200))
        self.assertTrue(m.release())


class MapPortOrder(unittest.TestCase):
    def test_falls_through_methods(self):
        order = []

        def fail(name):
            def f(*a, **k):
                order.append(name)
                raise nat.NatError(f"{name} no")
            return f

        ok = nat.Mapping("203.0.113.1", 50000, 80, "nat-pmp")
        pmp = lambda *a, **k: (order.append("nat-pmp"), ok)[1]
        got = nat.map_port(80, local_ip="192.168.1.2", gateway="192.168.1.1", upnp=fail("upnp"), pcp=fail("pcp"), pmp=pmp)
        self.assertIs(got, ok)
        self.assertEqual(order, ["upnp", "pcp", "nat-pmp"])

    def test_all_fail_reports_each(self):
        f = lambda *a, **k: (_ for _ in ()).throw(nat.NatError("nope"))
        with self.assertRaises(nat.MappingFailed) as cm:
            nat.map_port(80, local_ip="192.168.1.2", gateway="192.168.1.1", upnp=f, pcp=f, pmp=f)
        self.assertEqual(set(cm.exception.errors), {"upnp", "pcp", "nat-pmp"})

    def test_no_gateway_skips_gateway_methods(self):
        f = lambda *a, **k: (_ for _ in ()).throw(nat.NatError("nope"))
        with self.assertRaises(nat.MappingFailed) as cm:
            nat.map_port(80, local_ip="192.168.1.2", gateway="", upnp=f, pcp=f, pmp=f, methods=("pcp",))
        self.assertIn("pcp", cm.exception.errors)


class Gateways(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(nat.parse_route_get_default("   route to: default\n    gateway: 192.168.1.1\n  interface: en0\n"),
                         "192.168.1.1")
        self.assertIsNone(nat.parse_route_get_default("nothing"))
        table = "Iface\tDestination\tGateway\tFlags\nwlan0\t00000000\t0101A8C0\t0003\nwlan0\t0001A8C0\t00000000\t0001\n"
        self.assertEqual(nat.parse_proc_net_route(table), "192.168.1.1")

    def test_local_addresses_filters(self):
        got = nat.local_addresses(probe=lambda: ["127.0.0.1", "169.254.1.1", "192.168.1.5", "192.168.1.5", "::1", "bad"])
        self.assertEqual(got, ["192.168.1.5"])


class Reachability(unittest.TestCase):
    def test_direct_on_first_true(self):
        calls = []
        r = nat.check_reachability("203.0.113.1", 50000, lambda ip, p: calls.append((ip, p)) or len(calls) == 2)
        self.assertEqual((r.verdict, r.answers, r.direct), ("direct", [False, True], True))

    def test_unreachable_when_all_refuse_or_cannot_ask(self):
        r = nat.check_reachability("203.0.113.1", 1, lambda ip, p: False)
        self.assertEqual((r.verdict, r.answers), ("unreachable", [False] * 3))
        r = nat.check_reachability("203.0.113.1", 1, lambda ip, p: None, tries=2)
        self.assertIn("could be asked", r.detail)

        def boom(ip, p):
            raise RuntimeError("dht down")
        r = nat.check_reachability("203.0.113.1", 1, boom, tries=1)
        self.assertEqual((r.verdict, r.answers), ("unreachable", [None]))

    def test_tcp_dial_refuses_private_and_other_targets(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            self.assertFalse(nat.tcp_dial("127.0.0.1", port))      # loopback: never
            self.assertFalse(nat.tcp_dial("192.168.1.1", 80))      # private: never
            self.assertFalse(nat.tcp_dial("not-an-ip", 80))
            self.assertFalse(nat.tcp_dial("8.8.8.8", 80, requester_ip="1.2.3.4"))  # not the requester
        finally:
            srv.close()


class NeverHang(unittest.TestCase):
    def test_udp_exchange_times_out(self):
        quiet = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        quiet.bind(("127.0.0.1", 0))
        try:
            started = time.monotonic()
            self.assertIsNone(nat.udp_exchange(quiet.getsockname(), b"x", 0.6, tries=2))
            self.assertLess(time.monotonic() - started, 1.5)
        finally:
            quiet.close()

    def test_udp_exchange_ignores_other_senders_and_returns_reply(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(("127.0.0.1", 0))

        def echo():
            data, addr = server.recvfrom(100)
            server.sendto(b"re:" + data, addr)
        t = threading.Thread(target=echo, daemon=True)
        t.start()
        try:
            self.assertEqual(nat.udp_exchange(server.getsockname(), b"hi", 2.0), b"re:hi")
        finally:
            t.join(2)
            server.close()

    def test_resolve_times_out_on_slow_resolver(self):
        started = time.monotonic()
        self.assertIsNone(nat._resolve("slow.example", 0.3, resolver=lambda *a, **k: time.sleep(3)))
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(nat._resolve("1.2.3.4"), "1.2.3.4")

    def test_http_request_times_out(self):
        silent = socket.socket()
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        try:
            started = time.monotonic()
            with self.assertRaises(OSError):
                nat.http_request("GET", f"http://127.0.0.1:{silent.getsockname()[1]}/", timeout=0.4)
            self.assertLess(time.monotonic() - started, 2.0)
        finally:
            silent.close()

    def test_http_request_refuses_https(self):
        with self.assertRaises(nat.NatError):
            nat.http_request("GET", "https://192.168.1.1/")


class Report(unittest.TestCase):
    def test_full_report_keeps_no_mapping_by_default(self):
        mapped = []
        r = nat.full_report(stun=lambda **k: {"public_ip": "203.0.113.1", "mapped_port": 5, "nat_kind": "cone", "mappings": []},
                            local=lambda: ["192.168.1.5"], gateway=lambda: "192.168.1.1", discover=lambda: [],
                            pmp_probe=lambda g: None, mapper=lambda *a, **k: mapped.append(1))
        self.assertEqual(mapped, [])
        self.assertFalse(r["upnp"]["available"])
        self.assertFalse(r["nat_pmp"]["available"])
        self.assertNotIn("mapping", r)

    def test_try_map_releases(self):
        m = nat.Mapping("203.0.113.1", 50500, 1, "upnp")
        deleted = []
        m._delete = lambda: deleted.append(1)
        r = nat.full_report(try_map=True, stun=lambda **k: {"public_ip": None, "mapped_port": None, "nat_kind": "unknown", "mappings": []},
                            local=lambda: ["192.168.1.5"], gateway=lambda: None, discover=lambda: [],
                            pmp_probe=lambda g: None, mapper=lambda *a, **k: m)
        self.assertEqual(deleted, [1])
        self.assertTrue(r["mapping"]["deleted"])

    def test_redact(self):
        self.assertEqual(nat.redact_ip("203.0.113.9"), "203.x.x.x")
        self.assertIsNone(nat.redact_ip(None))


if __name__ == "__main__":
    unittest.main()
