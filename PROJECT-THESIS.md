# Common Network — project thesis and what we have built

This document is the single-page explanation of the project, the bet it is built around, and the current state of the system. It is written for anyone who needs to understand what is being tested and why, without reading the whole repository.

---

# 1. The thesis

**Claim:** A network of narrow, weight-specialised AI models running on consumer hardware can answer better than any single model on that network, when the network combines only the specialists that are each best at something different.

The opposite of this is the status quo: a handful of companies run enormous general-purpose models behind closed APIs, and everyone else rents access. Common is the counter-proposal — a permissionless network where anyone can contribute a specialist node, and requests are routed to the models that actually fit them.

Three ideas are packed into that claim:

1. **Specialists can beat generalists at narrow tasks.** A 7–8B model fine-tuned for maths or code can outperform a 70B general model on that task.
2. **A panel of non-dominated specialists can combine.** A maths specialist and a writing specialist, working on a prompt that needs both, can produce something neither could produce alone.
3. **A local network can run this on consumer hardware.** Laptops in a school lab can hold the models, talk over the LAN, and route/aggregate without a central cloud gatekeeper.

This repository is built to test whether claim (2) and (3) hold together.

---

# 2. Why this is not already settled

The v0.1 experiments tested composition and found **no gain**. That result is important, but it tested composition under the one condition where it cannot help:

- The two "specialists" were not really complementary. At 70–72B, the *writing* model was the better mathematician (0.615 vs 0.577). One model dominated the other on both axes.
- The pre-flight check found **zero maths-tuned models** on OpenRouter across all 411 model IDs. The experiment that could have tested the thesis could not be run, because the right model was not available on that platform.

The findings say this directly: *"This is not evidence that mixture-of-experts fails."* The blocker was procurement, not physics. Weight-specialised models do exist on `registry.ollama.ai` (`mathstral`, `qwen2.5-math`, `qwen2.5-coder`, etc.). They are free to host locally and unprofitable to host as a paid API. That gap **is** the thesis.

---

# 3. What we built

## 3.1 Common Network v0.1.1 — composition in the gateway

The gateway is no longer a load balancer. For requests that span domains, it:

1. **Plans a panel.** It picks specialists whose declared lanes match the request, and refuses to compose if one node dominates every matched domain.
2. **Fans out in parallel.** Each specialist gets the same prompt, rewritten for its lane.
3. **Verifies deterministically.** Arithmetic in specialist answers is re-derived in `Decimal`; cross-specialist disagreements are surfaced to the aggregator.
4. **Aggregates.** A generalist model synthesises one answer from the panel outputs and the verification report.
5. **Degrades safely.** If only one specialist answers, the user still gets a real answer — the strongest single specialist, not a failure.

The OpenAI-compatible API surface is unchanged. Existing clients keep working and get composition transparently. `X-Common-Compose: never` restores exact v0.1 single-node behaviour.

## 3.2 `common test --thesis` — the confirm/deny instrument

Added to `common/common.py`. This is the test to run on the school lab if the question is "does the thesis work?".

It runs four arms on deterministic ground-truth cases:

| Arm | What it measures |
|---|---|
| `single` | v0.1 routing — one request to one node. Baseline. |
| `panel` | Composition forced on. The network's actual claim. |
| `best-member` | Each panel member asked alone, best score kept. **The decisive control.** |
| `replication` | `single` run again. Noise floor. |

The cases deliberately span:

- **code + math** (write a function and return a number)
- **code + writing** (write a function and explain it)
- **math + writing** (explain a mathematical truth with an example)
- **debug + writing** (find a bug, fix it, describe the fix)
- **out-of-scope refusals** (security and medical prompts)

The verdict is `panel − best-member`. Beating `single` is weak — it only shows the panel beat whatever the router happened to pick. Beating the best individual member of the same panel is the actual thesis, and it is a much harder bar.

## 3.3 Real specialists in the catalogue

The catalogue is now built around narrow specialists rather than generalists. For the school-lab test, the relevant entries are:

| Lane | Catalogue ID | Ollama tag | Size |
|---|---|---|---|
| Math | `qwen2.5-math-7b` | `mightykatun/qwen2.5-math:7b` | ~8 GB |
| Math (light) | `qwen2.5-math-1.5b` | `mightykatun/qwen2.5-math:1.5b` | ~1.6 GB |
| Code | `qwen2.5-coder-7b` | `qwen2.5-coder:7b` | ~4.7 GB |
| Code (light) | `qwen2.5-coder-1.5b` | `qwen2.5-coder:1.5b` | ~1 GB |
| Writing / aggregate | `qwen3-8b` | `qwen3:8b` | ~4–5 GB |
| Writing / aggregate (light) | `llama3.2-3b` | `llama3.2:3b` | ~2 GB |

`verified_in_lane` is deliberately false for all of them. That flag is earned by measurement, not assumed. The thesis test is what will earn it.

## 3.4 LAN mode — the school-lab path

`common join --lan` lets machines join the network over the local subnet with no Cloudflare tunnel, no public hostname, and no perimeter firewall changes. It was built because v0.1's default tunnel is exactly what school networks block.

The only likely blocker on a school LAN is **client/AP isolation** — machines on the same subnet cannot reach each other. The runbook includes a one-line test for this.

## 3.5 Documentation

- `SCHOOL-NETWORK-REQUIREMENTS.md` — the IT handout.
- `SCHOOL-LAB-THESIS-RUNBOOK.md` — the full operational checklist.
- `CHANGELOG-v0.1.1.md` — detailed technical changes.
- `testing/seam-findings.md` — the v0.1 experiments and diagnosis.

---

# 4. How to run the decisive test

On ~10 school laptops, with Ollama and the Common CLI installed:

```bash
# 1. Gateway machine (one laptop)
cd common-network/gateway
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.migrate
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Each other laptop — start Ollama bound to the LAN
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 3. Each other laptop — join the network by lane
COMMON_NO_UPDATE=1 common join --lan \
  --gateway http://<gateway-ip>:8000 \
  --model qwen2.5-math-7b --auto        # maths machines

COMMON_NO_UPDATE=1 common join --lan \
  --gateway http://<gateway-ip>:8000 \
  --model qwen2.5-coder-7b --auto       # code machines

COMMON_NO_UPDATE=1 common join --lan \
  --gateway http://<gateway-ip>:8000 \
  --model qwen3-8b --auto               # aggregator machines

# 4. Verify the network can compose
common recommend --machines 10

# 5. Run the thesis test
COMMON_NO_UPDATE=1 common test \
  --gateway http://<gateway-ip>:8000 \
  --thesis --full \
  --out thesis.jsonl
```

The `--full` flag runs 3 repeats per arm so the noise floor is tight enough to trust.

---

# 5. What the result means

The report compares `panel` against `best-member`. Three outcomes:

| Outcome | Interpretation |
|---|---|
| `panel` > `best-member` beyond the noise floor | **Thesis SUPPORTED** for these cases/models. The network of specialists answers better than any single one. |
| `panel` ≈ `best-member` within the noise floor | **Thesis INCONCLUSIVE**. The panel is not adding anything, but not destroying value either. The specialists may not be genuinely non-dominated. |
| `panel` < `best-member` beyond the noise floor | **Thesis DENIED** for these cases/models. The right engineering choice is to route to the best single specialist and skip aggregation. |

The report also checks **refusal loss**: if the panel answers out-of-scope questions that a single specialist would refuse, composition has destroyed the specialist's main advantage.

---

# 6. Current honest status

- **The network runs.** v0.1.1 has been validated: the gateway starts, composition forms panels, verification runs, and degradation works.
- **The thesis test instrument exists and is self-tested.** `common test --thesis` has been run against mock nodes end-to-end.
- **It has not been run against live models yet.** That is the school-lab run.
- **No claim is being made that composition works.** v0.1.1 is a hypothesis-testing machine, not a product claiming victory.

The next step is the live run. This repository is ready for it.
