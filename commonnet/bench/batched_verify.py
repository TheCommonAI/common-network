"""python -m commonnet.bench.batched_verify : how does one verify pass cost grow with concurrent users?

COM-28. A verify pass checks k draft tokens for each user in one forward pass, so it processes
users x (k+1) positions. If the pass is memory-bound (it reads the weights once whatever the batch),
4 users should cost about what 1 user costs. This script measures that with llama.cpp's own
`llama-batched-bench` (the same release archive `common join` already fetches), on whatever backend
that binary is built for: pass `--ngl` / `--rpc` / `--device` to run it across GPU or Metal nodes.

Per user count it reports the median pass time (`T_PP`, the prompt-shaped phase that stands in for a
verify pass), tokens per pass, ms per user, and the ratio to one user. It also reports the KV cache
each active user needs, per model copy and per stage, from the model shape llama.cpp logs.

    python -m commonnet.bench.batched_verify --model model.gguf --users 1,2,4,8 --k 5 \
        --llama-bin /path/to/llama-batched-bench --ngl 99 --rpc host1:50052,host2:50052 --stages 3

Limits, stated so the numbers are not over-read:
- `llama-batched-bench` starts each sequence at an empty context, so the pass runs at shallow depth.
  Attention cost at long context is not measured here; `--ctx-per-seq` sets the cache size per user.
- A pipeline across RPC stages adds hop latency that this single-invocation bench does not include.
  Stage time is measured where llama.cpp runs the whole pass; treat it as compute, not end-to-end latency.

Verdict heuristic (ratio = verify ms at 4 users / verify ms at 1 user, or at the largest user count
when 4 is not in the list):
  <= 1.25 : about the same -> batching holds, queueing is mostly solved.
  >= 0.75 * users : linear -> plan for multiple model copies earlier.
  otherwise : partly batched; report the table and the ratio as they are.
Writes the table to --out as JSON when given.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)
MODEL_INFO = re.compile(r"print_info:\s+(n_layer|n_head_kv|n_embd_head_k)\s*=\s*(\d+)")
LINEAR_RATIO = 0.75
FLAT_RATIO = 1.25


def parse_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        pp, tg, users, n_kv, t_pp = m.group(1, 2, 3, 4, 5)
        rows.append({"pp": int(pp), "tg": int(tg), "users": int(users), "n_kv": int(n_kv),
                     "t_pp_s": float(t_pp)})
    return rows


def parse_model_info(text: str) -> dict:
    info: dict[str, int] = {}
    for name, value in MODEL_INFO.findall(text):
        info.setdefault(name, int(value))
    missing = {"n_layer", "n_head_kv", "n_embd_head_k"} - info.keys()
    if missing:
        raise ValueError(f"llama.cpp log did not report {sorted(missing)}; run with -v")
    return info


def kv_bytes_per_token_per_layer(info: dict, bytes_per_elem: float) -> float:
    # K and V, each n_head_kv x n_embd_head_k values per layer per token.
    return 2 * info["n_head_kv"] * info["n_embd_head_k"] * bytes_per_elem


def kv_report(info: dict, ctx_per_seq: int, stages: int, bytes_per_elem: float) -> dict:
    per_token_layer = kv_bytes_per_token_per_layer(info, bytes_per_elem)
    total = per_token_layer * info["n_layer"] * ctx_per_seq
    return {
        "kv_bytes_type": bytes_per_elem,
        "ctx_per_seq": ctx_per_seq,
        "layers": info["n_layer"],
        "stages": stages,
        "per_user_total_mib": round(total / 2**20, 2),
        "per_user_per_stage_mib": round(total / stages / 2**20, 2),
    }


def verdict(table: list[dict]) -> str:
    base = next((r for r in table if r["users"] == 1), None)
    top = next((r for r in table if r["users"] == 4), None) or max(table, key=lambda r: r["users"])
    if base is None or top["users"] == 1:
        return "need a 1-user row and a larger user count to judge"
    ratio = top["verify_ms_median"] / base["verify_ms_median"]
    if ratio <= FLAT_RATIO:
        return f"{top['users']} users cost {ratio:.2f}x one user: about the same, batching holds"
    if ratio >= LINEAR_RATIO * top["users"]:
        return f"{top['users']} users cost {ratio:.2f}x one user: roughly linear, plan for multiple model copies"
    return f"{top['users']} users cost {ratio:.2f}x one user: partly batched, see table"


def run_once(args, users_list: list[int], extra: list[str]) -> tuple[list[dict], str]:
    ctx = max(users_list) * args.ctx_per_seq
    cmd = [args.llama_bin, "-m", args.model, "-c", str(ctx),
           "-npp", str(args.k + 1), "-ntg", "1", "-npl", ",".join(map(str, users_list)),
           "-t", str(args.threads), "-v"] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"llama-batched-bench failed ({proc.returncode}):\n{text[-2000:]}")
    return parse_rows(text), text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="GGUF model (the same weights every stage loads)")
    ap.add_argument("--llama-bin", default="llama-batched-bench", help="path to llama-batched-bench")
    ap.add_argument("--users", default="1,2,4,8", help="comma-separated concurrent user counts")
    ap.add_argument("--k", type=int, default=5, help="draft tokens per user; each pass is k+1 positions")
    ap.add_argument("--reps", type=int, default=3, help="measured repetitions (median is reported)")
    ap.add_argument("--ctx-per-seq", type=int, default=128, help="KV cells reserved per user")
    ap.add_argument("--stages", type=int, default=1, help="pipeline stages the layers are split across")
    ap.add_argument("--kv-bytes", type=float, default=2.0, help="bytes per KV element (f16=2, q8_0~1.06)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ngl", help="GPU layers to offload (passed through, e.g. 99)")
    ap.add_argument("--rpc", help="comma-separated RPC servers host:port (passed through)")
    ap.add_argument("--device", help="devices to offload to (passed through)")
    ap.add_argument("--out", help="write the JSON result here")
    args = ap.parse_args(argv)

    users_list = sorted({int(u) for u in args.users.split(",") if u.strip()})
    extra: list[str] = []
    if args.ngl is not None:
        extra += ["-ngl", str(args.ngl)]
    if args.rpc:
        extra += ["--rpc", args.rpc]
    if args.device:
        extra += ["--device", args.device]

    samples: dict[int, list[dict]] = {u: [] for u in users_list}
    info = None
    # First run is a warm-up: weights page in and the first pass pays one-time costs.
    for rep in range(args.reps + 1):
        rows, text = run_once(args, users_list, extra)
        info = info or parse_model_info(text)
        if rep == 0:
            continue
        for r in rows:
            samples[r["users"]].append(r)

    table = []
    for u in users_list:
        if not samples[u]:
            raise RuntimeError(f"no measurement for {u} users")
        times = [r["t_pp_s"] * 1000 for r in samples[u]]
        med = statistics.median(times)
        table.append({
            "users": u,
            "tokens_per_pass": u * (args.k + 1),
            "verify_ms_median": round(med, 3),
            "verify_ms_min": round(min(times), 3),
            "verify_ms_max": round(max(times), 3),
            "ms_per_user": round(med / u, 3),
        })
    base = next((r for r in table if r["users"] == 1), None)
    for r in table:
        r["vs_1_user"] = round(r["verify_ms_median"] / base["verify_ms_median"], 3) if base else None

    kv = kv_report(info, args.ctx_per_seq, args.stages, args.kv_bytes)
    result = {"model": Path(args.model).name, "k": args.k, "reps": args.reps, "backend_args": extra,
              "model_info": info, "kv": kv, "table": table, "verdict": verdict(table)}

    print(f"{'users':>5} {'tokens/pass':>12} {'verify ms':>10} {'min':>9} {'max':>9} {'ms/user':>9} {'vs 1':>7}")
    for r in table:
        print(f"{r['users']:>5} {r['tokens_per_pass']:>12} {r['verify_ms_median']:>10.2f} "
              f"{r['verify_ms_min']:>9.2f} {r['verify_ms_max']:>9.2f} {r['ms_per_user']:>9.2f} {r['vs_1_user']:>7.2f}")
    print(f"KV per active user: {kv['per_user_total_mib']} MiB total, "
          f"{kv['per_user_per_stage_mib']} MiB per stage ({kv['stages']} stage(s), {kv['ctx_per_seq']} cells)")
    print("verdict:", result["verdict"])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
