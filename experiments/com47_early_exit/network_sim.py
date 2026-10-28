"""COM-47 network projection: pipelined early-exit drafting with asynchronous verification.

The client drafts blocks from the first L layers and keeps drafting while earlier blocks are verified remotely
(up to M blocks in flight). A block is verified by the full model; accepted tokens are committed; on a mismatch,
in-flight blocks are discarded and drafting restarts from the verifier's correction token. Acceptance flags come
from the measured teacher-forced records (feasibility.py cache), so acceptance is real. Timing uses measured CPU
step costs and an assumed round-trip time.
"""

import argparse
import collections
import json
import pickle
from pathlib import Path

C_FULL = 0.0477
VERIFY_PER_EXTRA_TOKEN = 0.00854
DRAFT_STEP_MEASURED = 0.0425


def verify_time(n_tokens):
    return C_FULL + VERIFY_PER_EXTRA_TOKEN * (n_tokens - 1)


def simulate(c, pmax, K, M, tau, D, rtt):
    G = len(c)
    t = 0.0
    j = 0
    f = 0
    server_free = 0.0
    inflight = collections.deque()
    while j < G:
        if inflight and inflight[0]["ret"] <= t:
            pass
        elif len(inflight) < M and f < G:
            n = 0
            steps = 0
            while n < K and f + n < G:
                steps += 1
                if tau is not None and pmax[f + n] < tau:
                    break
                n += 1
            if n == 0 and inflight:
                t = max(t, inflight[0]["ret"])
            else:
                t += steps * D
                arrive = max(t + rtt / 2, server_free)
                finish = arrive + verify_time(n + 1)
                server_free = finish
                inflight.append({"start": f, "n": n, "ret": finish + rtt / 2})
                f += n
                continue
        head = inflight[0]
        t = max(t, head["ret"])
        n, start = head["n"], head["start"]
        a = 0
        while a < n and c[start + a]:
            a += 1
        if a == n and n > 0 and M > 1:
            inflight.popleft()
            j = min(start + n, G)
            continue
        commit = a + 1 if a < n else n + 1
        inflight.clear()
        j = min(start + commit, G)
        f = j
    return j, t


def run(records, layer, K, M, tau, D, rtt):
    tokens = 0
    time_total = 0.0
    for rec in records:
        d = rec["layers"][layer]
        c = (d["d"] == rec["y"]).tolist()
        pmax = d["pmax"].tolist()
        tok, tm = simulate(c, pmax, K, M, tau, D, rtt)
        tokens += tok
        time_total += tm
    return tokens / time_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/com47/records_q06.pkl")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--out", default="experiments/com47_early_exit/network_sim_results.json")
    args = ap.parse_args()

    with open(args.cache, "rb") as fh:
        records = pickle.load(fh)

    rtt = 0.086
    plain = 1.0 / (rtt + C_FULL)
    rows = [{"config": "plain (1 round trip per token)", "tok_s": round(plain, 2), "speedup": 1.0}]
    for D, label in ((DRAFT_STEP_MEASURED, "measured CPU draft step 42.5 ms"),
                     (0.015, "illustrative cheap draft step 15 ms")):
        for K in (4, 8):
            for M, mode in ((1, "synchronous"), (2, "pipelined M=2"), (4, "pipelined M=4")):
                for tau, gname in ((None, "ungated"), (0.5, "pmax>=0.5 gate")):
                    tps = run(records, args.layer, K, M, tau, D, rtt)
                    rows.append({
                        "draft": label, "K": K, "mode": mode, "gate": gname,
                        "tok_s": round(tps, 2), "speedup": round(tps / plain, 3),
                    })
    for row in rows:
        print(json.dumps(row))

    sens = []
    for rtt_ms in (20, 50, 86, 150, 250):
        r = rtt_ms / 1000
        p = 1.0 / (r + C_FULL)
        best = max(run(records, args.layer, 4, M, None, 0.015, r) for M in (2, 4))
        sync = run(records, args.layer, 4, 1, None, DRAFT_STEP_MEASURED, r)
        sens.append({"rtt_ms": rtt_ms, "plain_tok_s": round(p, 2),
                     "sync_measured_draft_tok_s": round(sync, 2),
                     "pipelined_cheap_draft_tok_s": round(best, 2)})
        print(json.dumps(sens[-1]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"rtt_ms_default": 86, "layer": args.layer, "rows": rows, "rtt_sensitivity": sens}, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
