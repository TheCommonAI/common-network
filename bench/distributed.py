#!/usr/bin/env python3
"""Measure the actual v0.2 RPC runtime, including drafting and verification.

Starts each arm independently. Peers must already be running. Does not register
with a gateway or write node identity. Results include generated text locally.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "join"))
from runtime import DistributedConfig, FetchFailed


def request(url, body=None, timeout=240):
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}), timeout=timeout)


def generate(base, model, prompt, max_tokens):
    start = time.monotonic()
    times, text, usage = [], [], {}
    finished = False
    with request(base + "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "seed": 0, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                finished = True
                break
            item = json.loads(raw)
            if item.get("error"):
                raise RuntimeError(str(item["error"]))
            usage = item.get("usage") or usage
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    times.append(time.monotonic())
                    text.append(content)
    elapsed = time.monotonic() - start
    if not finished:
        raise RuntimeError("runtime stream ended before [DONE]; generation may be incomplete")
    if not times:
        raise RuntimeError("runtime returned no generated content")
    count = usage.get("completion_tokens")
    return {"seconds": elapsed, "ttft_seconds": times[0] - start,
            "max_chunk_gap_seconds": max((b - a for a, b in zip(times, times[1:])), default=0),
            "completion_tokens": count,
            "end_to_end_tokens_per_second": count / elapsed if count else None,
            "output": "".join(text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("prompts", type=Path, help="directory of .txt prompts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.tokens < 1 or args.repeats < 1:
        parser.error("port, tokens, and repeats must be positive and valid")
    config = DistributedConfig(args.config)
    if not config.draft:
        parser.error("benchmark config must specify a draft_model")
    prompts = sorted(args.prompts.glob("*.txt"))
    if not prompts:
        parser.error("no .txt prompts found")
    config.preflight()
    base = f"http://127.0.0.1:{args.port}"
    draft = config.draft
    # Exclusive creation prevents accidentally overwriting a previous run.
    with args.out.open("x") as output:
        for arm in (0, 4, 8, 12):
            config.draft = draft if arm else None
            config.draft_tokens = arm or 4
            with args.out.with_suffix(f".k{arm}.log").open("xb") as log:
                proc = subprocess.Popen(config.argv(args.port), stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 300
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f"runtime exited: {proc.returncode}; see {log.name}")
                        try:
                            with request(base + "/health", timeout=2):
                                pass
                            with request(base + "/props", timeout=2) as response:
                                props = json.load(response)
                            if Path(props.get("model_path", "")).resolve() != config.target:
                                raise RuntimeError("port belongs to a different model runtime")
                            # Do not accept an existing runtime that won the bind race.
                            time.sleep(0.2)
                            if proc.poll() is not None:
                                raise RuntimeError("new runtime failed to bind; choose another port")
                            break
                        except (urllib.error.URLError, OSError):
                            if time.monotonic() >= deadline:
                                raise RuntimeError("runtime startup timed out")
                            time.sleep(0.5)
                    generate(base, config.name, "Reply with one word: ready", 8)
                    for repeat in range(args.repeats):
                        for prompt in prompts:
                            result = generate(base, config.name, prompt.read_text(), args.tokens)
                            record = {"arm": "plain" if not arm else f"k={arm}",
                                      "prompt": prompt.name, "repeat": repeat,
                                      "rpc_servers": config.peers, "devices": config.devices,
                                      "context": config.context, "parallel": config.parallel, **result}
                            output.write(json.dumps(record) + "\n")
                            output.flush()
                            print(f"{record['arm']} {prompt.name}: {result['end_to_end_tokens_per_second']} tok/s", flush=True)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()


if __name__ == "__main__":
    try:
        main()
    except (FetchFailed, OSError, RuntimeError, ValueError) as e:
        sys.exit(str(e))
