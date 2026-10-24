# commonnet.dht: seedless peer discovery

A small Kademlia DHT over UDP (stdlib + `cryptography`). It stores the existing ed25519-signed peer
records from `commonnet/mesh/records.py`, so a peer can find the others with no seed URL. Nothing
imports it yet: it is off until an integration step constructs a `DhtNode`.

```python
node = DhtNode(identity, bind=("0.0.0.0", 0), bootstrap=load_bootstrap(), network_id="common-main-8b-v1")
node.start(); node.bootstrap()
node.announce(identity.sign_record({...}))   # call again every ~15 s
node.find_peers()                            # live, verified records for the network
node.lookup_peer(peer_id)                    # one peer's record by id
node.stop()
```

Run a plain node (a public bootstrap node), or list peers:
`python -m commonnet.dht --bootstrap HOST:PORT --network ID [--list]`. `--list` never prints tokens.

## Design
- Ids are sha256, 256 bit, XOR distance. Node id = sha256("node/" + peer id). Records live under two
  keys: `sha256("net/" + network_id)` (everyone in a network) and `sha256("peer/" + peer id)`. One DHT can
  carry many networks; a record is only accepted under the key that matches its own `network`/`peer`.
- k = 20 buckets, alpha = 3 parallel lookups, least-recently-seen first, replacement cache, a full
  bucket pings its oldest entry and evicts it only if it stays silent. Idle buckets are refreshed by a
  lookup (300 s). Two failed requests in a row drop a contact.
- `announce` finds the k closest nodes to each key and STOREs the record there. `find_peers` finds the
  closest nodes to the network key and pages FIND_VALUE through the 8 closest, verifies every record,
  keeps live ones (`is_live`) and the highest `seq` per peer.
- Stored values expire when the record is older than `LIVE_SECONDS` (60 s), so the caller re-announces.

## Messages
One JSON object per UDP datagram, at most 1200 bytes, common fields `t` (type), `n` (8 random bytes,
hex, echoed back) and `id` (sender peer id).

| request | extra fields | response |
| --- | --- | --- |
| `ping` | | `pong` |
| `fn` | `k` target (64 hex) | `nodes`: up to 16 `[peer, host, port]` |
| `fv` | `k` key, `a` return records with peer id > a | `vals`: records that fit, `more`, 3 `nodes` on the first page |
| `st` | `k` key, `tok` write token, `r` record | `ok`, or `err` with `why` |

Responses carry `pub` and `sig`: an ed25519 signature over the whole response (including the echoed
nonce) by the key whose hash is `id`. Responses also carry `tok`, a write token (HMAC of the requester's
IP, rotating every 5 min). Because one record is ~500 bytes, `fv` is paged, not fragmented.

## Defences
- Every record is checked with `verify_record`; refused: bad signature, older than 60 s, more than 120 s
  in the future, more than 800 encoded bytes, wrong key for its network/peer. Older `seq` never replaces newer.
- A contact enters the routing table only when it returns a response signed by the key behind its
  claimed id, from the address we sent to, echoing our nonce. Spoofed sources and made-up ids cannot get in.
  Unknown senders of requests are pinged (queue of 128, once per 30 s per address) before they count.
- STORE needs a token that only a node that really receives packets at its IP can have.
- Caps: 256 values per key, 128 values per source IP, token bucket per source address (100/s, burst 200)
  and per IP (1000/s). Malformed or oversized (>1500 B) packets are counted and dropped.
- Replies are never larger than max(300 B, the request), so a spoofed ping cannot be amplified more than
  ~4x. Requests that want a full page are themselves padded to ~1100 B.

## Not defended (be honest about it)
- Sybil and eclipse: node ids come from keys anyone can generate, so an attacker can place many nodes
  near the network key and answer every lookup (hiding peers or returning only stale ones). A record still
  cannot be forged, but it can be withheld. Mitigation to add later: bootstrap from several operators and
  cross-check with the gossip seed.
- Network-key hot spot: all peers of a network are stored on the same ~20 nodes, 256 values each. Fine for
  hundreds of peers, not for tens of thousands (shard the key then).
- Privacy: the records, including their `url` and `token`, are readable by any DHT node and by anyone who
  queries. Do not put secrets there that you would not gossip. Source IPs are visible to the nodes you talk to.
- No NAT traversal (another module), IPv4 only, no encryption of the transport, no clock-skew tolerance
  beyond the record limits, and a reflection attack on a victim is only bounded, not prevented.
- Replay of an old response is blocked by the nonce, but replay of a still-live record is by design
  possible (it carries no secret of its own and expires in 60 s).
