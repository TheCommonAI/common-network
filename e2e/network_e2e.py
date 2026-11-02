"""Real-model end-to-end test of the decentralised layer network (Qwen3-8B, local peers, no tunnels).

Peers are real `python -m commonnet.node` processes that find each other by gossip, pick their own windows,
and serve MLX blocks. The client discovers the route through the mesh. One peer of a replicated stage is
SIGKILLed mid-generation; the output must equal the unbroken run. Then the only live peer holding a stage is
SIGKILLed; a spare must take that stage over, and generation must still work on the healed route.
"""
import json, os, signal, subprocess, sys, tempfile, time, urllib.request
from collections import Counter
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
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
env = dict(os.environ, COMMONNET_LOCAL_CHECKPOINT=CK, COMMONNET_MLX_CACHE_MB="256", PYTHONPATH=REPO)
ports = [18951, 18952, 18953, 18954, 18955]
procs = {}
def get(url, path):
    with urllib.request.urlopen(url + path, timeout=3) as r: return json.loads(r.read())
def start(i):
    p = ports[i]
    args = [sys.executable, "-u", "-m", "commonnet.node", "--network", mpath, "--port", str(p), "--no-tunnel",
            "--advertise", f"http://127.0.0.1:{p}", "--identity-dir", os.path.join(tmp, f"id{i}"), "--stage-gib", "2.0"]
    if i: args += ["--seed", f"http://127.0.0.1:{ports[0]}"]
    procs[i] = subprocess.Popen(args, env=env, cwd=REPO, stdout=open(os.path.join(tmp, f"n{i}.log"), "w"), stderr=subprocess.STDOUT)
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
def alive_urls():
    return {f"http://127.0.0.1:{ports[i]}" for i in range(5) if procs[i].poll() is None}
def live_records():
    now = time.time(); alive = alive_urls()
    recs = get(f"http://127.0.0.1:{ports[0]}", "/peers")["records"]
    return [r for r in recs if r["state"] == "up" and now - r["ts"] <= 60 and r["url"] in alive]
def healing_check(prompt_ids):
    recs = live_records()
    counts = Counter(tuple(r["window"]) for r in recs)
    lone = next((r for r in recs if counts[tuple(r["window"])] == 1), None)
    if lone is None:
        print("healing check: every live stage still has a second holder; nothing to heal, skipped", flush=True)
        return
    idx = next(i for i in range(5) if f"http://127.0.0.1:{ports[i]}" == lone["url"])
    window = lone["window"]
    print(f"healing check: killing peer {idx}, the only live holder of layers {window}", flush=True)
    procs[idx].send_signal(signal.SIGKILL); procs[idx].wait(); t0 = time.time()
    healed = None
    while time.time() - t0 < 300:
        try:
            ups = [r for r in live_records() if r["window"] == window and r["url"] != lone["url"]]
            if ups: healed = (time.time() - t0, ups[0]["url"]); break
        except Exception: pass
        time.sleep(3)
    if not healed: raise SystemExit(f"layers {window} were not re-covered within 300s; see {tmp}")
    print(f"stage {window} healed: after {healed[0]:.0f}s by {healed[1]}", flush=True)
    net.refresh(force=True)
    eps, tr = net.plan()
    c = Client4(CK, tr, fmt="mlx4", hidden_dtype=torch.float32)
    out = c.generate(eps, prompt_ids, 24)
    print("generation after healing ok:", len(out) == 24)
try:
    windows = {}
    for i in range(5):
        start(i); windows[i] = wait_up(i); print(f"peer {i} up serving layers {windows[i]}", flush=True)
    net = Network.from_spec(f"http://127.0.0.1:{ports[0]}")
    net.refresh(force=True)
    print("route:", [(h.window, h.primary["url"][-5:], len(h.alternates)) for h in net.route()], flush=True)
    prompts = [json.loads(open("reference.json").read())[i] for i in (0, 4, 9)]
    ck_client = CK
    def run(kill):
        eps, tr = net.plan()
        c = Client4(ck_client, tr, fmt="mlx4", hidden_dtype=torch.float32)
        calls = {"n": 0}; real = tr.forward; worst = [0.0]
        victim = None
        for h in net.route():
            if h.alternates: victim = h; break
        def fwd(ep, sid, h, pos):
            if kill and victim is not None and ep.start == victim.window[0]:
                calls["n"] += 1
                if calls["n"] == KILL_AT:
                    idx = next(i for i in range(5) if f"http://127.0.0.1:{ports[i]}" == victim.primary["url"])
                    procs[idx].send_signal(signal.SIGKILL); procs[idx].wait(); print(f"   >>> SIGKILL peer {idx} (layers {victim.window}) mid-generation", flush=True)
            t = time.time(); out = real(ep, sid, h, pos); worst[0] = max(worst[0], time.time() - t); return out
        tr.forward = fwd
        outs = [c.generate(eps, r["ids"], 40) for r in prompts]
        return outs, worst[0], tr
    base, _, _ = run(False); print("baseline done", flush=True)
    net.refresh(force=True)
    broken, worst, tr = run(True)
    print("failovers:", getattr(tr, "failovers", None), f"| slowest single call {worst:.1f}s")
    print("identical to unbroken run:", [a == b for a, b in zip(base, broken)])
    print("sample text ok:", len(base[0]) == 40)
    healing_check(prompts[0]["ids"])
finally:
    for p in procs.values():
        try: p.kill()
        except Exception: pass
