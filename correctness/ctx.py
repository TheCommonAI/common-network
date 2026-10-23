"""Context-length sweep over two local block servers (or one with cut=n). Usage: ctx.py CKPT CUT LEN[,LEN...]"""
import json, os, secrets, subprocess, sys, time, tempfile, resource
import torch
sys.path.insert(0, "/Users/mileshedrick/commonnet-run")
from commonnet.client import BlockEndpoint
from commonnet.client4 import Client4
from commonnet.net.transport import HttpTransport
from commonnet.net.client import BlockClient
from commonnet.net import auth
from transformers import AutoTokenizer
ck, cut, lens = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
n = json.load(open(os.path.join(ck, "config.json")))["num_hidden_layers"]
tokfile = os.path.join(tempfile.mkdtemp(), "t"); open(tokfile, "w").write(secrets.token_urlsafe(32)); os.chmod(tokfile, 0o600)
env = dict(os.environ, COMMONNET_BACKEND="mlx", COMMONNET_CHECKPOINT=ck, COMMONNET_DTYPE="bfloat16")
spans = [(0, cut, 18821)] + ([(cut, n, 18822)] if cut < n else [])
procs = [subprocess.Popen([sys.executable, "-u", "-m", "commonnet.net.server", "--port", str(p), "--layers", str(a), str(b),
         "--block", "commonnet.net.factory:FACTORY", "--token-file", tokfile], env=env, cwd="/Users/mileshedrick/commonnet-run",
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for a, b, p in spans]
tok = AutoTokenizer.from_pretrained(ck)
text = open("/Users/mileshedrick/commonnet-run/README.md").read() * 40
allids = tok(text)["input_ids"]
res = {}
try:
    token = auth.load_token(tokfile); bc = BlockClient(token=token)
    for _, _, p in spans:
        for _ in range(120):
            try: bc.health(f"http://127.0.0.1:{p}", timeout=2); break
            except Exception: time.sleep(1)
    eps = [BlockEndpoint(f"m{i}", a, b) for i, (a, b, _) in enumerate(spans)]
    tr = HttpTransport({f"m{i}": f"http://127.0.0.1:{p}" for i, (_, _, p) in enumerate(spans)}, token=token)
    c = Client4(ck, tr, fmt="mlx4", hidden_dtype=torch.float32)
    for L in lens:
        ids = allids[:L]
        try:
            for cl in tr._clients.values(): cl.timings.clear()
            t0 = time.time(); out = c.generate(eps, ids, 17); wall = time.time() - t0
            per = [[t for t in cl.timings if t["path"] == "/forward"] for _, cl in sorted(tr._clients.items())]
            bigc = sum(t["server_ms"] or 0 for rows in per for t in rows if t["bytes_out"] > 60000) / 1000
            nbig = sum(1 for rows in per for t in rows if t["bytes_out"] > 60000); mb = sum(t["bytes_out"] for rows in per for t in rows if t["bytes_out"] > 60000) / 1e6
            big = sum(t["total_ms"] for rows in per for t in rows if t["bytes_out"] > 60000) / 1000
            steps = [sum(rows[i]["total_ms"] for rows in per) for i in range(len(per[0])) if per[0][i]["bytes_out"] <= 60000]
            comp = [sum(rows[i]["server_ms"] or 0 for rows in per) for i in range(len(per[0])) if per[0][i]["bytes_out"] <= 60000]
            import statistics as st
            res[L] = {"wall_s": round(wall, 1), "prefill_hops_s": round(big, 1), "decode_ms_median": round(st.median(steps)),
                      "decode_compute_ms_median": round(st.median(comp)), "decode_ms_max": round(max(steps)), "out": out}
            fp = subprocess.run("for p in $(pgrep -f 'commonnet.net.server --port 1882'); do footprint -p $p 2>/dev/null | grep Footprint | head -1 | sed 's/.*Footprint: //; s/ (.*//'; done", shell=True, capture_output=True, text=True).stdout.split()
            res[L]["server_footprint"] = fp
            print(f"len {L:6d}: servers {fp}  wall {wall:6.1f}s  prefill(hops) {big:6.1f}s [server compute {bigc:.1f}s, {nbig} calls, {mb:.0f} MB up]  decode/step median {st.median(steps):6.0f} ms "
                  f"(compute {st.median(comp):.0f}, max {max(steps):.0f})", flush=True)
        except Exception as e:
            res[L] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(f"len {L:6d}: FAILED {res[L]['error']}", flush=True); break
finally:
    for p in procs: p.terminate()
    json.dump(res, open(f"ctx_cut{cut}.json", "w"))
