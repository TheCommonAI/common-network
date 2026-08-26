# School-lab thesis test runbook

This is the operational checklist for running `common test --thesis` on ~10 school laptops and getting a clean confirm/deny result for the specialist-network thesis.

> **Use `Full Network Build v0.1.1/`**, not v0.1. v0.1 routes one request to one node and cannot test the thesis. v0.1.1 has composition, verification, LAN mode, and `common test --thesis`.

---

## What the test will prove

`common test --thesis` runs four arms on deterministic ground-truth cases:

| Arm | What it answers |
|---|---|
| `single` | v0.1 behaviour — one request to one node. Baseline. |
| `panel` | Composition forced on — the network's actual claim. |
| `best-member` | Each panel member asked alone; best score kept. **The decisive control.** |
| `replication` | `single` run again. Noise floor. |

The cases deliberately span **code + math** and **code + writing**, because that's where a real panel can win: the code specialist cannot write the essay, and the writing specialist cannot write the function.

The verdict is based on `panel − best-member`. If `panel` wins by more than the noise floor, the thesis is supported for those cases/models. If it loses, the thesis is denied and you should route to the best specialist instead.

---

## 1. What to ask IT

Use `SCHOOL-NETWORK-REQUIREMENTS.md` as the main handout. For the thesis test you specifically need:

- **LAN mode** (no Cloudflare tunnels). Each laptop must reach the others on the same subnet.
- **Outbound HTTPS (443)** to `registry.ollama.ai`, `*.ollama.com`, `github.com`, `raw.githubusercontent.com`, `pypi.org`, `files.pythonhosted.org` — for one-time installs.
- **Machine-to-machine TCP** on ports `11434` (Ollama) and `8000` (gateway).
- **Client/AP isolation disabled** for the lab subnet. Test with `curl http://<other-ip>:11434/api/tags` from one machine to another.
- **Admin rights** to install Ollama and run `common join`.
- **Machines stay awake** — disable sleep during the test.

**Optional / non-local path:** if you ever want the public tunnel path, you additionally need outbound access to `*.trycloudflare.com`, `*.argotunnel.com`, port `7844`, and `gateway-production-b820.up.railway.app`. Only ask for this if LAN mode is impossible — a tunnel exposes a school machine to the public internet and IT will rightly push back.

---

## 2. Model strategy for 10 laptops

Do not let every machine pull the same model. That produces N clones of one generalist and the network cannot beat its own best node.

### Recommended split

| Laptops | Role | Catalogue ID | Ollama tag | Approx size | RAM needed |
|---|---|---|---|---|---|
| 3–4 | Math specialist | `qwen2.5-math-7b` | `mightykatun/qwen2.5-math:7b` | ~8 GB | 16 GB |
| 1–2 | Math specialist (light) | `qwen2.5-math-1.5b` | `mightykatun/qwen2.5-math:1.5b` | ~1.6 GB | 8 GB |
| 3–4 | Code specialist | `qwen2.5-coder-7b` | `qwen2.5-coder:7b` | ~4.7 GB | 16 GB |
| 1–2 | Code specialist (light) | `qwen2.5-coder-1.5b` | `qwen2.5-coder:1.5b` | ~1 GB | 8 GB |
| 2–3 | Generalist / aggregator | `qwen3-8b` | `qwen3:8b` | ~4–5 GB | 16 GB |
| 0–2 | Generalist (light) | `llama3.2-3b` | `llama3.2:3b` | ~2 GB | 8 GB |

### Minimum viable panel

For the thesis test to run, the network needs **only three lanes live**:

1. **math** (any qwen2.5-math variant)
2. **code** (any qwen2.5-coder variant)
3. **general** / **writing** (any qwen3 / llama3.1 / llama3.2 variant, with `can_aggregate: true`)

The exact split of the 10 laptops is flexible. What matters is that at least two distinct specialist lanes are covered plus one aggregator.

### Bandwidth

Twenty machines pulling ~4 GB each = **~80 GB**. Do not do this simultaneously during class.

1. **Preferred:** Pull once on one machine, then copy `~/.ollama/models` to the others over the LAN.
2. **Fallback:** Pull the day before in small batches.

---

## 3. Pre-flight checklist

Run these on every machine before the main test:

```bash
# 1. Ollama is installed
curl -fsSL https://ollama.com/install.sh | sh

# 2. Ollama is running and bound to the LAN
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 3. This machine is reachable from others (run from a peer)
curl http://<this-machine-ip>:11434/api/tags

# 4. Install the Common CLI
curl -fsSL https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.sh | sh
```

If `curl http://<other-ip>:11434/api/tags` fails from a peer, **client isolation is on** — that is the single most common blocker.

---

## 4. Deploy the network

### Gateway machine (one laptop)

```bash
cd "common-network/gateway"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # edit DATABASE_URL if not using local Postgres
.venv/bin/python -m app.migrate
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Every other laptop

Start Ollama bound to the LAN:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Then join the network with a specific lane. Use `COMMON_NO_UPDATE=1` so the CLI does not overwrite the local `--thesis` changes before the run:

```bash
# Maths machines (16 GB)
COMMON_NO_UPDATE=1 common join --lan --gateway http://<gateway-ip>:8000 --model qwen2.5-math-7b --auto

# Maths machines (8 GB or nearly full)
COMMON_NO_UPDATE=1 common join --lan --gateway http://<gateway-ip>:8000 --model qwen2.5-math-1.5b --auto

# Code machines (16 GB)
COMMON_NO_UPDATE=1 common join --lan --gateway http://<gateway-ip>:8000 --model qwen2.5-coder-7b --auto

# Code machines (8 GB or nearly full)
COMMON_NO_UPDATE=1 common join --lan --gateway http://<gateway-ip>:8000 --model qwen2.5-coder-1.5b --auto

# Generalist / aggregator machines (16 GB)
COMMON_NO_UPDATE=1 common join --lan --gateway http://<gateway-ip>:8000 --model qwen3-8b --auto

# Generalist / aggregator machines (8 GB or nearly full)
COMMON_NO_UPDATE=1 common join --lan --gateway http://<gateway-ip>:8000 --model llama3.2-3b --auto
```

The `--model` value is the **catalogue ID**, not the raw Ollama tag. `common join` resolves it automatically.

---

## 5. Verify composition is possible

From any machine:

```bash
common recommend --machines 10
```

It should report **at least 2 distinct specialist lanes + 1 generalist** and `can_compose: true`. If it says the network cannot compose, fix the fleet before running the thesis test.

Also run a quick smoke test:

```bash
common ask --gateway http://<gateway-ip>:8000 --compose \
  "Write a Python function fib(n) and tell me fib(10)."
```

The response headers should show:

```
X-Common-Topology: panel
X-Common-Panel: mock-maths, mock-coder
X-Common-Aggregator: mock-generalist
```

If you see `X-Common-Topology: single`, check `X-Common-Compose-Reason` for why the panel did not form.

---

## 6. Run the thesis test

From one client machine:

```bash
COMMON_NO_UPDATE=1 common test \
  --gateway http://<gateway-ip>:8000 \
  --thesis --full \
  --out thesis-math-code.jsonl
```

`--full` runs 3 repeats per arm for a tighter noise floor. On a cold lab this may take 10–30 minutes.

---

## 7. Read the verdict

The terminal prints a table like this:

```
arm               mean  in-scope  refusal   chars      ms
--------------------------------------------------------------------
single           0.708     0.812      50%      53      15
panel            0.750     0.875      50%      50      20
best-member      1.000     1.000     100%      54      17
replication      0.708     0.812      50%      53      16

NOISE FLOOR ...

VERDICT
  panel − single         +0.042   REAL
  panel − best member    -0.250   REAL

  → Composition is WORSE than its best member...
```

Interpretation:

| Result | Meaning |
|---|---|
| `panel` beats `best-member` by more than the noise floor | **Thesis SUPPORTED** for these cases/models. |
| `panel` ≈ `best-member` (within noise floor) | **Thesis INCONCLUSIVE** — specialists are not adding value, but not destroying it. Check they are genuinely non-dominated. |
| `panel` loses to `best-member` beyond noise floor | **Thesis DENIED** — route to the best single specialist instead. |
| `panel` refuses fewer out-of-scope questions than `single` | **REFUSAL LOSS** — composition is destroying the specialist's main advantage. |

---

## 8. Capture and share results

The full results are in the JSONL file passed to `--out`. Keep it. To share a readable report:

1. Keep the JSONL.
2. Copy the terminal output or write it to a file.
3. Note which models were on which machines, RAM per machine, and any IT blockers you hit.

Do not quote a single-repeat run as final — run `--full` (3 repeats) or better.

---

## 9. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Refusing to run thesis test...only X specialist lane(s)` | Fleet is cloned generalists, or a required specialist is missing. | Reassign models; use `common recommend`. |
| Nodes register but `healthy: false` | Ollama not running, or bound to localhost only, or client isolation. | Set `OLLAMA_HOST=0.0.0.0:11434`; ask IT about client isolation. |
| Test runs but all scores are 0 | Models are returning garbage, or the ground-truth parser is failing. | Spot-check a few answers manually; verify expected figures appear. |
| `panel` never forms (topology always `single`) | Confidence gate or domination gate is blocking composition. | Check `X-Common-Compose-Reason` header; verify specialists are in different lanes. |
| Gateway can't start / migrations fail | Postgres not running or `pgvector` missing. | `createdb common_network; psql -d common_network -c "create extension vector;"` |
| CLI overwrites `--thesis` changes | `common` self-updates from GitHub. | Set `COMMON_NO_UPDATE=1` or use `--no-update`. |

---

## 10. Non-local path (if LAN is impossible)

Only use this if IT will not allow LAN mode.

Each laptop runs:

```bash
common join --gateway https://gateway-production-b820.up.railway.app --model <specialist> --auto
```

This opens a Cloudflare quick tunnel from the laptop to the public gateway. The laptop is reachable from the internet, which is why schools commonly block it. It requires:

- Outbound HTTPS to `*.trycloudflare.com`
- Outbound to `*.argotunnel.com` / Cloudflare edge on 443 and 7844
- `cloudflared` allowed to run
- Outbound HTTPS to `gateway-production-b820.up.railway.app`

If IT says no to any of these, fall back to LAN mode.
