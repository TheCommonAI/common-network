"""Kill the second-half member mid-generation; a replica takes over. Output must equal an unbroken run."""
import json, os, secrets, signal, subprocess, sys, time, tempfile
import torch
sys.path.insert(0, "/Users/mileshedrick/commonnet-run")
from commonnet.client import BlockEndpoint
from commonnet.client4 import Client4
from commonnet.net.transport import HttpTransport
from commonnet.net.client import BlockClient
from commonnet.net import auth
ck, kill_at = sys.argv[1], int(sys.argv[2])
n = json.load(open(os.path.join(ck, "config.json")))["num_hidden_layers"]; cut = n // 2
ref = json.load(open("reference.json"))[:3]
tokfile = os.path.join(tempfile.mkdtemp(), "t"); open(tokfile, "w").write(secrets.token_urlsafe(32)); os.chmod(tokfile, 0o600)
env = dict(os.environ, COMMONNET_BACKEND="mlx", COMMONNET_CHECKPOINT=ck, COMMONNET_DTYPE="bfloat16")
spans = {"A": (0, cut, 18831), "B1": (cut, n, 18832), "B2": (cut, n, 18833)}
procs = {k: subprocess.Popen([sys.executable, "-u", "-m", "commonnet.net.server", "--port", str(p), "--layers", str(a), str(b),
         "--block", "commonnet.net.factory:FACTORY", "--token-file", tokfile], env=env, cwd="/Users/mileshedrick/commonnet-run",
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for k, (a, b, p) in spans.items()}
url = lambda k: f"http://127.0.0.1:{spans[k][2]}"
try:
    token = auth.load_token(tokfile); bc = BlockClient(token=token)
    for k in spans:
        for _ in range(180):
            try: bc.health(url(k), timeout=2); break
            except Exception: time.sleep(1)
    eps = [BlockEndpoint("m0", 0, cut), BlockEndpoint("m1", cut, n)]
    def run(kill):
        tr = HttpTransport({"m0": url("A"), "m1": url("B1")}, replicas={"m1": [url("B2")]}, token=token)
        c = Client4(ck, tr, fmt="mlx4", hidden_dtype=torch.float32)
        calls = {"m1": 0}; real = tr.forward; slow = []
        def fwd(ep, sid, h, pos):
            if ep.name == "m1":
                calls["m1"] += 1
                if kill and calls["m1"] == kill_at:
                    procs["B1"].send_signal(signal.SIGKILL); procs["B1"].wait(); print("   >>> killed B1 at m1 call", kill_at, flush=True)
            t = time.time(); out = real(ep, sid, h, pos); slow.append(time.time() - t); return out
        tr.forward = fwd
        outs = [c.generate(eps, r["ids"], len(r["gen"])) for r in ref]
        return outs, tr, max(slow)
    base, _, _ = run(False)
    print("baseline done", flush=True)
    broken, tr, worst = run(True)
    same = [a == b for a, b in zip(base, broken)]
    print("failovers:", tr.failovers, f"| slowest single call {worst:.1f}s (includes detection, retry backoff and replay)")
    print("identical to unbroken run:", same)
finally:
    for p in procs.values():
        try: p.kill()
        except Exception: pass
