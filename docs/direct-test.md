# Direct mode: a real-world test plan (three machines)

Status: **never run over a real WAN.** Everything below is what to run to find out. The loopback end-to-end test
(`commonnet/tests/test_direct_e2e.py`) shows the code path works; it cannot show that your router, your ISP or
Windows' firewall let it through. Expect to find at least one surprise. Design: `commonnet/net/DIRECT.md`;
short user doc: "Direct mode (experimental)" in `docs/layer-network.md`.

Ports used here (chosen to stay clear of 7460 and 11437): **TCP 7471** block server, **UDP 7472** the peer's DHT port,
**UDP 7480** an optional separate bootstrap node.

Goal: no Cloudflare and no seed URL. Anyone who knows one DHT address joins, serves, and uses anyone online.

## 0. What you need to know first

- **A public bootstrap must exist** for a stranger to join cold. Until one is listed in `commonnet/networks/bootstrap.json`
  (empty today) every joiner needs `--bootstrap HOST:PORT` of someone already in. The first reachable peer can be that
  address (its `--dht-port`), or run the stand-alone node below on any always-on machine.
- A peer is directly usable only if the **internet can open a TCP connection to it**. Home routers need a port forward;
  schools, offices and mobile hotspots usually cannot. Those peers fall back to the tunnel and still work.
- Machine A's router has **no UPnP / PCP / NAT-PMP**, so the forward is manual. Do not expect the automatic mapping to help.

## 1. Machine A: your Mac (192.168.20.41, behind the router, no UPnP)

**1a. Find the public IP** (from the Mac): `curl -s https://api.ipify.org` (say it prints `PUB.IP`). If the router's WAN
address differs from that, you are behind carrier-grade NAT and no forward will work: stop and use the tunnel.

**1b. Reserve the Mac's LAN address** in the router (DHCP reservation for 192.168.20.41) so the forward keeps pointing at it.

**1c. Add two port forwards** in the router admin page (names vary: "Port forwarding", "Virtual server", "NAT"):

| Protocol | External port | Internal IP | Internal port | Why |
|---|---|---|---|---|
| TCP | 7471 | 192.168.20.41 | 7471 | block server (TLS) |
| UDP | 7472 | 192.168.20.41 | 7472 | the peer's DHT port |

Do **not** forward the existing 7460 / 11437 ports for this test.

**1d. macOS firewall:** System Settings > Network > Firewall: if on, allow incoming connections for the Python you run
(`~/.common-network/runtime-venv/bin/python`), or turn the firewall off for the test.

**1e. Start the peer** (it is also the bootstrap for the others, via its UDP port 7472):

```
common join --direct --direct-port 7471 --external-port 7471 --public-ip PUB.IP --dht-port 7472
# or, without the launcher:
python -m commonnet.node --direct --port 7471 --external-port 7471 --public-ip PUB.IP --dht-port 7472
```

Success looks like (in this order):

```
DHT on udp 7472 (0 bootstrap node(s))
no bootstrap node and no seed: this peer starts the network; others join with --bootstrap <this machine's address>:7472
direct address decision: manual PUB.IP:7471 (no remote peer could check the forwarded port; published unverified)
serving layers 0-27 as <peer>
up at https://PUB.IP:7471
```

"published unverified" is expected for the first peer (nobody can dial back yet). From a machine **outside your LAN**
(a phone hotspot, or machine B) confirm both forwards before going on:

```
nc -vz PUB.IP 7471            # TCP: "succeeded"
python -m commonnet.net.nat   # on machine A: shows STUN public IP, NAT kind and (read-only) UPnP/PMP presence
```

(UDP is hard to test with `nc`; the proof is machine B finding A in step 2.)

**Hairpin NAT:** from the Mac itself, the router often will not connect to its own public address. That is fine: a node
talks to itself over loopback. To test through the public address, use machine B or a phone hotspot.

## 2. Machine B: Patrick's Windows PC (RTX 4060)

On Patrick's router do the same forward for his PC's LAN IP (`ipconfig` shows it): **TCP 7471** and **UDP 7472**.
Allow them through Windows Defender Firewall (PowerShell as Administrator):

```
netsh advfirewall firewall add rule name="Common block server" dir=in action=allow protocol=TCP localport=7471
netsh advfirewall firewall add rule name="Common DHT" dir=in action=allow protocol=UDP localport=7472
```

Patrick's public IP: `(Invoke-WebRequest https://api.ipify.org).Content`. Then, with `A.PUB.IP` from step 1a:

```
common join --direct --bootstrap A.PUB.IP:7472 --direct-port 7471 --external-port 7471 --public-ip B.PUB.IP --dht-port 7472
```

If his router has UPnP, you can omit `--public-ip/--external-port` and let it map and verify automatically; the log says
`the router mapped TCP ...` or `no automatic port mapping`.

Success: `direct address decision: manual B.PUB.IP:7471` **without** "unverified" (A dialled back), then `up at https://B.PUB.IP:7471`.
Step 4 shows A, B and C together.

## 3. Machine C: the school laptop (probably no inbound at all)

Nothing to forward. Run:

```
common join --direct --bootstrap A.PUB.IP:7472
```

Expected: it finds no router mapping, takes the STUN address, the dial-back fails, and it prints

```
direct address decision: unreachable ... (every remote peer failed to connect ...)
not reachable directly (...); using a Cloudflare tunnel instead. To serve directly, forward TCP ... 
opening a Cloudflare tunnel...
```

and then serves through the tunnel as today. It stays in the same mesh (its record travels by gossip and the DHT-found
peers' gossip). To prove the refusal path: add `--no-tunnel-fallback` and it exits with an error naming the port to forward.
(If the school network blocks outbound UDP, it may not even reach the DHT; then it is a plain tunnel peer, as before.)

## 4. Is it one mesh? How to see the route kind per hop

Anywhere (no URL, just a bootstrap):

```
python -m commonnet.dht --bootstrap A.PUB.IP:7472 --network common-main-8b-v1 --list
```

prints one line per live peer: `url=https://IP:PORT` is a **direct** peer, `https://xxx.trycloudflare.com` a **tunnel** peer.

Per hop of an actual route, and the benchmark with no URL:

```
python -m commonnet.bench --live --bootstrap A.PUB.IP:7472 --prompt-lens 16,256 --new-tokens 32
```

The route block prints one line per hop with its kind:

```
route: 3 stage(s), 3 live peer(s) known
  hop 0: layers 0-27   peer 59890399  mlx    direct B.PUB... 
  hop 1: layers 27-36  peer ab12cd34  cpu    tunnel https://....trycloudflare.com
```

The JSON (`bench-results/live-*.json`) has `route[].peers[].kind` and `per_hop[].kind`. For a chat client without a
URL: `python -m commonnet.serve --bootstrap A.PUB.IP:7472` (then `common ask` / the desktop app as usual).

What "success" means, in order of strength:

1. B and C appear in A's `--list` output (DHT works across the WAN, UDP forwards correct).
2. `bench --live` from a fourth place builds a route with at least one `direct` hop (TLS pinning works over the WAN).
3. Kill B's process mid-generation: a replica (or C) takes over and the output is identical (failover over direct peers).
4. Restart A with a different `--external-port` than is forwarded: it must say `unreachable` and fall back or refuse, not publish.

## 5. Stand-alone bootstrap node (optional, recommended)

On any always-on, reachable machine (a VPS with UDP 7480 open is ideal; a laptop that sleeps is not):

```
python -m commonnet.dht --bind 0.0.0.0:7480
```

Keep it running (systemd / launchd / a Scheduled Task). Its public `IP:7480` is what everyone passes to `--bootstrap`.
To make it the default for all joiners, put it in `commonnet/networks/bootstrap.json` on `main` (`["IP:7480"]`); until then
nobody can join without being told an address. One bootstrap is a single point of failure for **cold starts only**: once a
peer has joined it keeps a routing table, and more bootstrap addresses (repeat `--bootstrap`) remove the dependency.

## 6. What this test cannot tell you / likely failure modes

| Symptom | Likely cause |
|---|---|
| `every remote peer failed to connect` on a machine you forwarded | wrong internal IP, firewall on the host, or hairpin: test from outside; or ISP CGNAT (public IP differs from the router's WAN IP) |
| `no remote peer could be asked` on B | UDP 7472 not forwarded on A, or A's firewall drops UDP, or wrong `--bootstrap` address |
| B sees A in `--list` but A never sees B | A is reachable but B's UDP is not: B can still serve (records go through A), but only A-side lookups work |
| `PinMismatch` / `Unreachable` for a direct hop | the peer restarted with a new certificate: wait one announce cycle (15 s); the pin comes from the newest record |
| Works on Wi-Fi, not on a phone hotspot | hotspot is CGNAT: correct, that machine is `unreachable` |
| Slow first token on a direct hop | TLS handshake plus a cold TCP connection over the WAN; later calls reuse the connection |
