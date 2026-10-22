"""Split run: two local block servers (real HTTP, real client) vs reference.json."""
import json, os, secrets, subprocess, sys, time, tempfile
import torch
sys.path.insert(0, "/Users/mileshedrick/commonnet-run"); sys.path.insert(0, ".")
from commonnet.client import BlockEndpoint
from commonnet.client4 import Client4
from commonnet.net.transport import HttpTransport
from commonnet.net.client import BlockClient
from commonnet.net import auth
ck, cut = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(os.path.join(ck, "config.json")))
n = cfg.get("num_hidden_layers") or cfg["text_config"]["num_hidden_layers"]
ref = json.load(open("reference.json"))
tokfile = os.path.join(tempfile.mkdtemp(), "t"); open(tokfile, "w").write(secrets.token_urlsafe(32)); os.chmod(tokfile, 0o600)
MODE = sys.argv[3] if len(sys.argv) > 3 else "fp16"
env = dict(os.environ, COMMONNET_BACKEND="mlx", COMMONNET_CHECKPOINT=ck)
if MODE == "bf16": env["COMMONNET_DTYPE"] = "bfloat16"
spans = [(0, cut, 18811)] + ([(cut, n, 18812)] if cut < n else [])
procs = [subprocess.Popen([sys.executable, "-u", "-m", "commonnet.net.server", "--port", str(p), "--layers", str(a), str(b),
         "--block", "commonnet.net.factory:FACTORY", "--token-file", tokfile], env=env, cwd="/Users/mileshedrick/commonnet-run",
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for a, b, p in spans]
try:
    token = auth.load_token(tokfile)
    bc = BlockClient(token=token)
    for _, _, p in spans:
        for _ in range(120):
            try: bc.health(f"http://127.0.0.1:{p}", timeout=2); break
            except Exception: time.sleep(1)
        else: raise SystemExit("server did not come up")
    eps = [BlockEndpoint(f"m{i}", a, b) for i, (a, b, _) in enumerate(spans)]
    tr = HttpTransport({f"m{i}": f"http://127.0.0.1:{p}" for i, (_, _, p) in enumerate(spans)}, token=token)
    c = Client4(ck, tr, fmt="mlx4", **({"hidden_dtype": torch.float32} if MODE == "bf16" else {}))
    exact = 0; agree = total = 0; lines = []; gens = []
    for r in ref:
        t0 = time.time()
        gen = c.generate(eps, r["ids"], len(r["gen"])); gens.append(gen)
        match = gen == r["gen"]; exact += match
        first = next((i for i, (x, y) in enumerate(zip(gen, r["gen"])) if x != y), None)
        # teacher-forced: logits over prompt + reference continuation in one pass
        seq = torch.tensor([r["ids"] + r["gen"]])
        top1 = c.logits(eps, seq)[0].argmax(-1).tolist()
        a = sum(x == y for x, y in zip(top1, r["top1"])); agree += a; total += len(top1)
        lines.append(f"{'MATCH ' if match else 'DIFF  '} first_diff={first} tf_top1={a}/{len(top1)} {time.time()-t0:.1f}s  {r['prompt'][:40]!r}")
        print(lines[-1], flush=True)
    json.dump(gens, open(f"gens_cut{cut}_{MODE}.json", "w"))
    print(f"\nSPLIT at layer {cut} of {n}: greedy exact {exact}/{len(ref)}; teacher-forced top-1 {agree}/{total} = {100*agree/total:.2f}%")
    print("per-hop stats:", tr.stats())
finally:
    for p in procs: p.terminate()
