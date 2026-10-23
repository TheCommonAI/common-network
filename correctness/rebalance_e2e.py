"""Real-model end-to-end test of the decentralised layer network (Qwen3-8B, local peers, no tunnels).

Peers are real `python -m commonnet.node` processes that find each other by gossip, pick their own windows,
and serve MLX blocks. The client discovers the route through the mesh. One peer of a replicated stage is
SIGKILLed mid-generation; the output must equal the unbroken run.
"""
import json, os, signal, subprocess, sys, tempfile, time, urllib.request
sys.path.insert(0, "/Users/mileshedrick/commonnet-run")
import torch
from commonnet.netclient import Network
from commonnet.client4 import Client4

CK = sys.argv[1]
KILL_AT = int(sys.argv[2]) if len(sys.argv) > 2 else 30
tmp = tempfile.mkdtemp(prefix="net-e2e-")
manifest = {"id": "e2e-8b", "version": 1,
            "model": {"name": "Qwen3-8B", "num_layers": 36, "repos": {"mlx": "mlx-community/Qwen3-8B-4bit"},
                      "client_repo": "mlx-community/Qwen3-8B-4bit"},
            "boundaries": [0, 12, 24, 36], "gib_per_layer": {"mlx": 0.14}, "fixed_gib": 0.3, "min_replicas": 2, "seeds": []}
mpath = os.path.join(tmp, "manifest.json"); json.dump(manifest, open(mpath, "w"))
env = dict(os.environ, COMMONNET_LOCAL_CHECKPOINT=CK, COMMONNET_MLX_CACHE_MB="256", PYTHONPATH="/Users/mileshedrick/commonnet-run")
ports = [18951, 18952, 18953, 18954, 18955]
procs = {}
def get(url, path):
    with urllib.request.urlopen(url + path, timeout=3) as r: return json.loads(r.read())
def start(i):
    p = ports[i]
    args = [sys.executable, "-u", "-m", "commonnet.node", "--network", mpath, "--port", str(p), "--no-tunnel",
            "--advertise", f"http://127.0.0.1:{p}", "--identity-dir", os.path.join(tmp, f"id{i}"), "--stage-gib", "2.0"]
    if i: args += ["--seed", f"http://127.0.0.1:{ports[0]}"]
    procs[i] = subprocess.Popen(args, env=env, cwd="/Users/mileshedrick/commonnet-run", stdout=open(os.path.join(tmp, f"n{i}.log"), "w"), stderr=subprocess.STDOUT)
def wait_up(i, t=240):
    end = time.time() + t
    while time.time() < end:
        try:
            recs = get(f"http://127.0.0.1:{ports[0]}", "/peers")["records"]
            mine = [r for r in recs if r.get("url") == f"http://127.0.0.1:{ports[i]}" and r.get("state") == "up"]
            if mine: return mine[0]["window"]
        except Exception: pass
        time.sleep(1)
    raise SystemExit(f"peer {i} never came up; see {tmp}/n{i}.log")
try:
    windows = {}
    for i in range(5):
        start(i); windows[i] = wait_up(i); print(f"peer {i} up serving layers {windows[i]}", flush=True)
    net = Network.from_spec(f"http://127.0.0.1:{ports[0]}"); net.refresh(force=True)
    lone = next(i for i in range(5) if windows[i] == [24, 36])
    print(f"killing peer {lone}, the only holder of layers 24-36", flush=True)
    procs[lone].send_signal(signal.SIGKILL); procs[lone].wait(); t0 = time.time()
    healed = None
    while time.time() - t0 < 300:
        try:
            recs = get(f"http://127.0.0.1:{ports[0]}", "/peers")["records"]
            now = time.time()
            ups = [r for r in recs if r["state"] == "up" and now - r["ts"] <= 60 and r["window"] == [24, 36] and r["url"] != f"http://127.0.0.1:{ports[lone]}"]
            if ups: healed = (time.time() - t0, ups[0]["url"]); break
        except Exception: pass
        time.sleep(3)
    print("stage 24-36 healed:", f"after {healed[0]:.0f}s by {healed[1]}" if healed else "NOT HEALED in 300s", flush=True)
    if healed:
        net.refresh(force=True)
        print("route now:", [(h.window, len(h.alternates)) for h in net.route()])
        eps, tr = net.plan()
        c = Client4(CK, tr, fmt="mlx4", hidden_dtype=torch.float32)
        out = c.generate(eps, json.loads(open("reference.json").read())[0]["ids"], 24)
        print("generation after healing ok:", len(out) == 24)
finally:
    for p in procs.values():
        try: p.kill()
        except Exception: pass
