# Automatic peer connectivity

Run `common join`. No router login, port forwarding, or firewall editing is part
of the normal joining flow.

Peers keep their existing HTTPS address for discovery, a small authenticated
connection handshake, and fallback. In the background, compatible peers exchange
a WebRTC offer and answer at `/rtc/offer`, then perform ICE/STUN connectivity
checks. Both sides send outbound UDP traffic to establish a path through NAT.
The selected data channel uses DTLS encryption and reliable, ordered delivery.
Model traffic then goes directly between peers when that channel is ready.

The tunnel initially carries RPCs while the direct connection is negotiated. It
remains the fallback when UDP is blocked or NAT traversal fails. The handshake
does not introduce another relay service or send model weights to the gateway.
Some networks require relaying; automatic operation does not mean every pair can
always connect directly.

Token IDs, speculative verification, and grouped int8 activations keep their
existing RPC format. Changing between HTTP and the direct channel does not change
the model protocol or discard session state. A lost direct reply can retry the
same request over HTTPS, using the server's existing idempotency rules.

## Controls and diagnostics

- `common join --no-direct` keeps HTTP/tunnels only.
- `COMMONNET_RTC=0` disables automatic WebRTC upgrades for a process, including
  peer forwarding from that process.
- Advanced `--direct` options still support explicitly reachable TLS listeners.
  These are optional; ordinary users should not need them.
- `BlockClient.timings` records `transport: "webrtc"` only for RPCs that actually
  completed over a warm data channel, or `transport: "http"` for HTTPS/HTTP.
  An advertised tunnel URL alone therefore does not identify the model-data path.

The runtime installs the tested `aiortc==1.15.0` dependency automatically on an
upgrade. It is used for data channels only. Direct connections are scoped to the
transport lifecycle and closed when an API generation completes or is cancelled;
the initial request on a fresh transport may use HTTPS during negotiation.

Signaling uses the same bearer authentication and HTTPS certificate validation or
peer certificate pin as the existing RPC connection. Candidate descriptions are
bounded and restricted to public IP literals in ordinary operation; loopback and
private candidate support is an explicit local-test option. SDP and ICE secrets
are not written to timing reports or logs. Peers still see the prompts and model
activations they process.

## Verification limits

Tests cover actual local data-channel RPCs with the HTTP data path disabled after
negotiation, unchanged retries after a dropped reply, cleanup, and fallback to old
or unavailable peers. These checks do not establish throughput or compatibility
between every pair of home networks. A real two-location test is still required
before claiming that a particular WAN path is direct.

Protocol references: [ICE, RFC 8445](https://www.rfc-editor.org/rfc/rfc8445.html)
and the [aiortc API](https://aiortc.readthedocs.io/en/latest/api.html).
