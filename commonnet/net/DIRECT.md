# Direct connectivity (no Cloudflare tunnel)

Status: wired in behind `--direct` (experimental, off by default). `commonnet/direct.py` decides the address,
`commonnet/node.py` serves and announces, `commonnet/netclient.py` discovers and pins. Everything below marked
"not built" is not built. Verified so far: loopback processes in CI (`commonnet/tests/test_direct_e2e.py`) and unit
tests with fakes; never over a real WAN. The `ssl_context` / `pin` parameters still default to `None`.

| File | What it does |
|---|---|
| `commonnet/net/nat.py` | STUN, UPnP IGD, PCP, NAT-PMP, local addresses, reachability check, and a CLI |
| `commonnet/net/secure.py` | per-peer self-signed TLS certificate, fingerprint binding to the ed25519 identity, pinning client context |
| `commonnet/net/server.py` | `BlockServer(..., ssl_context=None)`: TLS, and (only with TLS and a token) a non-loopback bind |
| `commonnet/net/client.py` | `BlockClient(..., pin=None, ssl_context=None)`: certificate pinning for `https://IP:PORT` endpoints |

## The flow, as built

1. `Node.start()` joins the DHT (`commonnet.dht.DhtNode`, UDP, same ed25519 identity, network id from the manifest) using
   `--bootstrap`, `commonnet/networks/bootstrap.json` and `COMMONNET_DHT_BOOTSTRAP`, and merges what it finds into its
   peer table (so window placement sees DHT-only peers).
2. `_serve_direct`: `secure.make_peer_cert` (kept in the identity directory, so the pinned fingerprint is stable),
   `BlockServer("0.0.0.0", port, token, ssl_context)`.
3. `direct.decide()` picks the address: manual forward (`--public-ip`/`--external-port`), else `nat.map_port`, else
   STUN public IP + local port. It then asks DHT peers to dial back (`DhtNode.dial_back`, below) via
   `nat.check_reachability`. Verdict `direct` publishes `https://IP:PORT`; `unreachable` stops the TLS server and runs
   the existing tunnel path, or raises with the port to forward when `--no-tunnel-fallback`. The decision is made once;
   a window move keeps the address. Mappings are released on shutdown.
4. The 15 s re-sign loop signs the record (now with `fp`, `fpsig`), merges it into the gossip table, and announces it
   into the DHT. Gossip endpoints (`/peers`, `/gossip`, `/manifest`) are served on the same TLS port, so direct and
   tunnel peers interoperate; direct peers gossip with pinned TLS.
5. Clients (`Network.open`): `--network` omitted or `dht` uses the built-in manifest (the network id is the DHT key);
   `refresh()` asks the DHT and gossip. Every record is verified (`records.verify_record`); the pin map
   (`"ip:port" -> fp`) feeds `BlockClient(pin=...)` and is updated on every refresh and failover.
6. Every 120 s a direct node re-asks two peers to dial back and logs a warning if its address stopped answering.

### The dial-back RPC (`dial`, in `commonnet/dht/node.py`)

The requester sends a request signed with its identity (`pub`, `sig`) carrying the fresh per-IP token a previous reply gave it
(the same anti-spoofing token `STORE` uses), `ip` and `port`. The helper dials only the address the datagram came from,
and only if it equals `ip`, via `nat.tcp_dial` (global addresses only; private, loopback and link-local refused), at most
4 at a time, then answers `dialed {ok}`. A spoofed source cannot get a dial because it cannot present the token. It is a
TCP connect, not a TLS handshake.

### Record fields

`fp` and `fpsig` are optional fields of the signed record (`records.OPTIONAL_FIELDS`); `SIGNED_FIELDS` and
`RECORD_VERSION` are unchanged. `canonical()` already signs every key but `sig`, and `check_shape` ignores unknown keys, so a
node that predates this change **still verifies, stores and relays these records unchanged** (tested, including the DHT
size budget: `REQ_PAD` went from 1100 to 1190 so a record with the fields always fits one reply). What an old node
cannot do is use a direct peer: it connects without the pin and certificate verification fails, so it skips that peer.
New nodes enforce: `fp` and `fpsig` come together, the binding must verify against the record's own key, and an
`https://<IP>` URL without a valid binding is refused. Tunnel and plain records carry neither field.

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

## What works and what does not

| Situation | Result |
|---|---|
| Home router with UPnP / PCP / NAT-PMP enabled, public IPv4 | Works: mapping, dial-back, `direct` |
| Router without those protocols, user can forward a port by hand | Works: `--public-ip` / `--external-port` (see docs/direct-test.md) |
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

**A. Keep the Cloudflare tunnel as the fallback (recommended default).** Implemented: the node tries direct first, falls back
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
