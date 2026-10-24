# Direct connectivity (no Cloudflare tunnel)

Status: building blocks only. Nothing in the launcher calls them yet; a later integration step wires them in behind
an opt-in flag. Existing behaviour is unchanged: the new `ssl_context` / `pin` parameters default to `None`.

| File | What it does |
|---|---|
| `commonnet/net/nat.py` | STUN, UPnP IGD, PCP, NAT-PMP, local addresses, reachability check, and a CLI |
| `commonnet/net/secure.py` | per-peer self-signed TLS certificate, fingerprint binding to the ed25519 identity, pinning client context |
| `commonnet/net/server.py` | `BlockServer(..., ssl_context=None)`: TLS, and (only with TLS and a token) a non-loopback bind |
| `commonnet/net/client.py` | `BlockClient(..., pin=None, ssl_context=None)`: certificate pinning for `https://IP:PORT` endpoints |

## The flow the integration step should follow

1. `cert, key, fp = secure.make_peer_cert(identity, state_dir)`; `sig = secure.bind_fingerprint(identity, fp)`.
2. `BlockServer(block, "0.0.0.0", port, token=token, ssl_context=secure.server_ssl_context(cert, key))`.
3. `m = nat.map_port(server.port)` (UPnP, then PCP, then NAT-PMP). It renews itself; call `m.release()` on shutdown.
   If it raises `MappingFailed`, the router will not open a port: use the tunnel fallback.
4. `report = nat.check_reachability(m.external_ip, m.external_port, dial_back)` where `dial_back(ip, port)` asks a
   DHT peer to run `nat.tcp_dial(ip, port, requester_ip=<the address the request came from>)` and answers the
   result. `report.verdict` is `direct` or `unreachable`. Believe the dial-back, not the router: the router's
   "external IP" can be a private address behind carrier-grade NAT, and some routers accept a mapping they never forward.
5. If `direct`, publish `url = "https://IP:PORT"` and `fp` in the signed record. Else fall back to the existing tunnel.
6. Clients: `BlockClient(pin=record["fp"])` (or `pin={"IP:PORT": fp}` for many peers) after checking the record's
   signature with `records.verify_record`.

If there is no port mapping but STUN says `open` (a public address on this machine), skip step 3.

## What `secure.py` guarantees, and what it does not

* The TLS certificate is self-signed ECDSA P-256, stored as `peer-cert.pem` / `peer-key.pem` (key mode 0600) and reused
  until 30 days before expiry (10-year validity), so the fingerprint other peers have pinned stays stable.
* The client trusts exactly one certificate: the one whose sha256 (of the DER) equals the pin. No CA, no hostname or
  expiry check. A mismatch raises `PinMismatch` and the client does not retry it (an impostor will not get better).
* `bind_fingerprint` signs `b"commonnet-tls-cert-v1\n" + fp` with the identity key. `verify_binding(pub, fp, sig)`
  checks it. The prefix keeps this signature from being usable as a record signature or anything else.
* The pin is only as trustworthy as the channel that delivered it: a signed record (itself found through the DHT).
  A DHT peer that serves a forged record cannot produce a valid signature for someone else's identity.
* Plain HTTP to the TLS port fails (the handshake is rejected and the connection closed); the server does the handshake
  in the connection's own thread with a 10 s limit, so a stalled client cannot block accepting others.
* A non-loopback bind is refused unless a TLS context and a token are both given. The bearer token still guards
  `/health`, `/session`, `/forward`, ... exactly as before; `extra_handlers` stay public (signed content).

### Proposed `records.py` change (NOT made here)

`canonical()` already signs every key except `sig`, so a record that carries `fp` has it covered by the record
signature today. What is missing is (a) `sign_record` accepting it and (b) a shape check. Backwards compatible,
`SIGNED_FIELDS` and `RECORD_VERSION` unchanged (old records stay valid):

```python
OPTIONAL_FIELDS = ("fp",)
# sign_record: after building `record`:
#     if fields.get("fp"): record["fp"] = secure.normalize_fingerprint(fields["fp"])
# check_shape: after the SIGNED_FIELDS loop:
#     fp = record.get("fp")
#     if fp is not None:
#         try: secure.normalize_fingerprint(fp)
#         except ValueError: raise RecordError("fp must be a sha256 fingerprint")
```

Also allow `url` to be `https://<ipv4>:<port>` (it already is: any http(s) URL under 512 chars). Old peers that
do not know `fp` ignore the key but would connect without a pin and fail certificate verification, which is safe.

## What works and what does not

| Situation | Result |
|---|---|
| Home router with UPnP / PCP / NAT-PMP enabled, public IPv4 | Works: mapping, dial-back, `direct` |
| Router without those protocols, user can forward a port by hand | Works if the user forwards the port and tells the peer its public port (integration step) |
| Machine with a public IPv4 (cloud VM, some ISPs) | Works with no mapping; STUN reports `open` |
| Carrier-grade NAT (mobile, many ISPs), double NAT | Cannot be reached inbound. The router may even report a mapping with a private "external" address; dial-back fails, verdict `unreachable` |
| School, office, hotel networks that block inbound or UPnP | Cannot be reached. These peers can still act as clients (outbound works) but cannot serve a block without a relay |
| Symmetric NAT (STUN ports differ per destination) | A UPnP/PCP mapping is still a fixed port and can work. Without a mapping, hole punching is not possible, so unreachable |
| Cone NAT, no mapping available | Reachable only by UDP hole punching, which this TCP/HTTP design does not use |

STUN cannot tell full, restricted and port-restricted cone apart (that needs CHANGE-REQUEST, which the public servers do
not serve). We only need "same mapping for different destinations", which two servers prove.

IPv6: not handled yet. Parsing IPv6 STUN addresses is supported, but the probe, local-address and mapping code are IPv4.
With IPv6 there is usually no NAT but a stateful firewall; the correct tool is PCP (which supports IPv6 pinholes) or
UPnP `WANIPv6FirewallControl`. A public IPv6 address on a host with no inbound firewall is directly reachable, and
`https://[addr]:port` already works in the client. Worth adding after the IPv4 path is proven.

macOS note: SSDP multicast needs the "Local Network" permission for the terminal app. If it is denied, UPnP discovery
sees nothing and looks the same as "no UPnP router".

Security notes on router protocols: SSDP replies are untrusted LAN input. We only follow `LOCATION` and control URLs that
point at private LAN IPv4 addresses (no hostile redirect to the internet), refuse XML with a DTD, cap HTTP bodies at 256 KiB,
and time out every call (2 to 3 s). The only mapping created is TCP to this machine's LAN address, with a lease that is renewed.

## Relay fallback: design only (no code)

When the verdict is `unreachable`, two options:

**A. Keep the Cloudflare tunnel as the fallback (recommended default).** The integration step tries direct first, falls back
to `tunnel.py` for that peer, and publishes whichever URL works. No new protocol, no new abuse surface, but it keeps a
dependency on Cloudflare for exactly the peers that most need help.

**B. Circuit relay by directly reachable peers (later).** A peer R with verdict `direct` offers to forward for peer U
that is not reachable.

Minimal protocol:
1. U keeps an outbound TLS connection to R (`wss`-like long-lived HTTP/1.1 upgrade on R's block-server port, pinned by R's `fp`,
   authenticated by U's signed record). Outbound connections work from behind any NAT.
2. U's record carries `relay: {peer: R_id}` and `url = R's url + "/relay/" + U_peer`. R's signed record advertises `relay_slots`.
3. A client C sends the normal block-server requests to that URL. R tags each request with an id and writes it down U's connection;
   U answers on the same connection; R returns the reply. TLS between C and R only; to keep activations private from R, C
   additionally encrypts the body end to end to U's ed25519-derived X25519 key (the existing record has `pub`).
4. R enforces limits per U and per C: bytes per second, concurrent streams, total bytes per hour, request size (the existing
   `max_request_bytes`).

Abuse risks:
* Bandwidth theft: a relay carries other people's activation traffic (hundreds of KB per token). Needs quotas, an opt-in setting
  with a visible cap, and refusal when the host is metered.
* Open proxy: R must only forward to the one registered U connection, never to arbitrary host:port, and only the five block
  endpoints, never raw TCP.
* Confidentiality and integrity: R sees plaintext unless end-to-end encrypted; a malicious R can drop, delay or corrupt. Corruption
  is detected by the end-to-end layer; dropping shows as `Unreachable` and the coordinator reroutes.
* Denial of service: unauthenticated clients can exhaust R's slots. Require C's bearer token (U's record token) before R reads bodies.
* Traffic analysis and IP exposure: the relay learns who talks to whom. Document it.
* Amplification and reflection: `tcp_dial`-style helpers must only dial the requester's own address.
* Sybil relays: a relay can be run by an attacker to censor a peer. Peers should list two relays and prefer relays with history.

## Tests

`commonnet/tests/test_net_secure.py` (pin helpers; TLS end to end on loopback: right pin, wrong pin, plain HTTP, token,
keep-alive, a 700 KB forward, concurrency, stalled handshake) and `commonnet/tests/test_net_nat.py` (STUN packets, NAT kind,
UPnP XML and a fake IGD, PCP and NAT-PMP packets, mapping order, reachability, timeouts). No test touches the network
or a real router. `python -m commonnet.net.nat [--try-map] [--redact]` is the manual check on a real network.
