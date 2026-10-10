# Can specialists be blended on demand from public LoRA adapters?

An experiment, run on a MacBook Air (16 GB, MPS), asking one falsifiable question:

> Does a **LoraHub-style blend of two or more public LoRA adapters** answer better, on a
> domain genuinely in between the domains they were trained on, than **the base model** or
> **the single best adapter**?

Everything here exists to answer that, and the answer is allowed to be no. This is Stage 1
of a two-stage plan; Stage 2 — wiring on-demand adapter fusion into the gateway — is gated
on this experiment clearing a 5-point threshold, and is not built unless it does.

**What this does not claim.** Nothing here says the network gives better answers. It says
whether *this* blending procedure does, on *this* base model, on *these* questions.

---

## What is being compared

| | configuration | how it is chosen |
|---|---|---|
| **(a)** | base model | `Qwen/Qwen2.5-1.5B-Instruct`, untouched |
| **(b)** | best single adapter | whichever of the seven adapters scores highest on the 50 maths questions |
| **(c)** | the blend | LoraHub weights, searched on 20 held-out questions only |

(b) is deliberately chosen with hindsight, on the same questions the blend is then judged
on. It is therefore the strongest available baseline — it already knew the answers —
and a weaker one would only have flattered the blend.

---

## The adapters

Seven public LoRA adapters for `Qwen/Qwen2.5-1.5B-Instruct`, all pinned to a commit in
`adapters.json` so a re-run months from now loads the same bytes:

| domain | repo | blendable |
|---|---|---|
| maths | `mdhamidhosen/baamr-math-qwen2.5-1.5b-lora-12k` | yes |
| maths | `mdhamidhosen/baamr-math-qwen2.5-1.5b-lora-pilot-577` | yes |
| maths | `disha20041005/adaanchor-qwen2.5-1.5b-lora-math-k2` | yes |
| code | `bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16` | yes |
| code | `bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16v3` | yes |
| reasoning | `DarkyMan/Qwen2.5-1.5B-Opus46-Reasoning-LoRA` | yes |
| medical | `Arthur-77/QWEN2.5-1.5B-medical-finetuned` | **no** |

**Blendable** is decided by one thing: whether the adapter can be averaged in factor
space. LoraHub sums the `A` and `B` matrices across adapters and takes the product of the
sums, so every adapter in a blend must agree on `r`, `lora_alpha` and `target_modules`.
Six agree exactly (`r=16`, `alpha=32`, the same seven projections, ~70.5 MB each). The
medical adapter is `r=64`, `alpha=16` with four targets, so it is evaluated as a single
adapter and excluded from blends. `harness.load_registry` asserts this precondition on
every run rather than trusting the registry file.

### There is no maths-and-physics adapter, because none exists

The obvious in-between domain for this base would be physics. No physics or science
adapter exists for `Qwen2.5-1.5B` at any download count, so the in-between domain tested
is **maths word problems**, on a base blended from maths and code adapters. Every maths
item needs multi-step arithmetic, which is what the code adapters are nominally good at.

---

## The question sets

Both sets are generated, and both sets' answers are **computed rather than asserted**, so
a question and its answer cannot silently disagree.

- **Maths (50)** — ten families: unit price and change, discount then tax, rate/time/
  distance, averages, ratio sharing, compound growth, work rate, percentage increase then
  decrease, mixture concentration, and arithmetic series. Numbers come from a seeded RNG;
  every answer is computed in `fractions.Fraction` and quantised half-up, so it is exact.
- **Control (50)** — eight families rendered from encoded data tables: element symbols,
  atomic numbers, capitals, currencies, continents, SI units, planetary order, chemical
  formulae.
- **Held out (20 = 18 maths + 2 control)** — used *only* to choose the blend weights, and
  never scored.

`check_questions.py` runs 1037 assertions over all three files before anything is scored on
them: counts, no duplicate prompts, every maths answer re-derived from the item's own
parameters, **every parameter proven load-bearing** by perturbing it and requiring the
answer to move, every parameter's value shown to appear in the question text, control
answers checked against a fresh table lookup, held-out items proven absent from both scored
sets, and even family coverage.

### A generated set can be self-consistent and still be wrong

Worth writing down, because it is the failure this design is most exposed to.

The first generated maths set contained this item:

> A shop sells markers at £5.20 each. Alex buys 6 of them and pays with a £10 note. How
> much change, in pounds, does Alex receive?
> **Answer: −21.20**

Every one of the checks above passed on it. The answer re-derived from the parameters, both
parameters were load-bearing, both appeared in the question. It was simply an impossible
question — the generator had chosen the note by mixing pounds and pence, so £10 was offered
for £31.20 of goods.

Re-derivation cannot catch a wrong premise; it only proves the arithmetic agrees with
itself. It was caught by printing one item per family and reading them, which is why that
step is in the plan and why `check_questions.py` now also asserts every maths answer is a
positive quantity — the one check here that is not about internal consistency. That check
was then verified to actually fire on the historical bug, since a check that only passes on
good data proves nothing.

---

## Method

- One base model loaded once. Adapters and blends are applied as a **forward-pass delta**
  through a `LoRALinear` wrapper, not merged into the weights — see below.
- Same chat template, same instruction, same `max_new_tokens` for every configuration. No
  per-configuration prompt tuning.
- **Greedy decoding** everywhere. A sampled run would put a random variable inside a
  comparison of a few points, and greedy makes the whole run reproducible.
- **One answer extractor**, fixed in advance and never revisited after seeing which
  configuration it favours: the last number in the completion for maths, compared with a
  tolerance of half a unit in the item's own stated decimal place; a word-boundary phrase
  match for the control set.
- Blend weights are searched on the held-out 20 only, under a **fixed candidate budget**,
  gradient-free — anchors (each single adapter, the uniform blend), then random draws on
  the simplex, then coordinate moves. Ties are broken toward *sparser* weights, which is
  the conservative direction: on a flat objective it collapses the blend toward a single
  adapter and can only cost the blend accuracy.

### Why there is no PEFT here

Applying a LoRA is `W_eff = W + (alpha/r) * B @ A`, and a LoraHub blend is the same
expression with `A` and `B` replaced by the weighted sums. That is tensor arithmetic, so
`blend.py` does it in plain `torch` + `safetensors`.

The reason is dependency risk, not purity: `transformers` is at 5.8.0 here, a major
release, and `peft` is the package most likely to break against it. With PEFT out of the
scoring path, `transformers` is only asked to load Qwen2.5 and generate — the stable part.

The saving is real beyond dependencies. A blend is ~74 MB of matrices rather than a ~3 GB
copy of the model; swapping configuration is 196 pointer assignments; returning to the
base model is `enabled = False` rather than a reload; and configuration N+1 cannot be
contaminated by configuration N, because the base weights are never written to.

Note the size: a six-way blend is **not** smaller than a single adapter — it is the same
74 MB. The blend has the same rank and shape as its inputs, because the sums `ΣλᵢAᵢ` and
`ΣλᵢBᵢ` are each still rank r, so it costs exactly what one adapter costs no matter how
many are blended in. The 74 MB is measured, not estimated.

The path is not merely asserted to be equivalent. `blend.py` writes the blend out in PEFT's
own layout and then checks, before the scored run, that applying the written file and
applying the in-memory matrices give **identical next-token distributions**.

That check is not decoration, and this is worth recording because it is the one place in
this experiment where a defensive check earned its keep. The LoraHub sum `ΣwᵢAᵢ` was
originally written out twice — once in the wrapper that applies a blend, once in the writer
that saves one — and the two copies drifted: the wrapper omitted the weight on its first
term, so it computed `A₀ + Σ_{i>0} wᵢAᵢ` where it should have computed `Σᵢ wᵢAᵢ`. For a
six-way blend that is not a rounding difference; it is a different model.

Nothing downstream could see it. The search ran to completion, every candidate produced a
plausible score, and the artifact was written correctly from the correct formula. The only
symptom was that the matrices on disk and the matrices being scored disagreed — which is
exactly what the logits comparison tests, and it failed the first run with a max logit
difference of 3.36 against a required 1e-3. The run stopped before scoring a single question,
so no number in `results.json` was ever produced by the wrong path.

The fix was not to correct the copy but to delete it: the sum now exists once, as
`harness.weighted_factor_sums`, and both callers use it. A model-free equivalence check
(`blend.check_apply_matches_write`) runs as a pre-flight before anything expensive, so the
same class of fault now surfaces in the first second rather than after a fifty-minute weight
search. It was verified to fail on the original buggy formula, since a check that has only
ever seen working code has not been tested.

---

## Results

Two runs, and they are the same experiment twice. The first fitted blend weights on 20
held-out items and scored everything on 50. It read +12.0 points over the base model while
its own paired interval still spanned zero, which is the case the pre-registered rule says
must stop rather than resolve itself. The second took those weights **frozen**, ran no
search at all, and scored three configurations on **200 fresh questions** that neither the
search nor the choice of baseline had ever seen.

Both numbers are below, because the second is only interpretable against the first.

### The run that stopped — 50 questions

| | configuration | accuracy | correct |
|---|---|---|---|
| **(c)** | **blend** | **0.420** | 21/50 |
| (b) | `math-adaanchor` — best single | 0.340 | 17/50 |
| | `math-12k` | 0.320 | 16/50 |
| **(a)** | **base model** | **0.300** | 15/50 |
| | `code-r16` | 0.300 | 15/50 |
| | `reasoning` | 0.280 | 14/50 |
| | `math-pilot` | 0.260 | 13/50 |
| | `medical` *(single-only)* | 0.260 | 13/50 |
| | `code-r16v3` | 0.140 | 7/50 |

| comparison | delta | exact McNemar | bootstrap CI | discordant |
|---|---|---|---|---|
| (c) − (a) blend vs base | +12.0 pts | p = 0.146 | [−0.020, +0.240] | 12 |
| (c) − (b) blend vs best single | +8.0 pts | p = 0.388 | [−0.060, +0.220] | 12 |

**Control set (50), three configurations:** base 0.760 (38/50), blend 0.740 (37/50),
`math-adaanchor` 0.700 (35/50). The blend cost one question of general ability against the
base and gained two against the single adapter. At n=50 that is a spread of one or two
items and should not be read as a difference either way; it is reported because the
treatment *could* have cost general ability and did not visibly do so.

**Verdict: STOP.** The point estimate cleared the 5-point rule against both baselines, and
the paired evidence did not support it. Under the rule above that is not a win and not a
loss — it is the case that comes back to be decided, which is what happened.

The weights were the output of 27 gradient-free evaluations against the 20 held-out items,
scoring 11/20 there (0.55), which took 4,936 s — 82 minutes of search to fit three numbers
on twenty questions. The blend artifact itself took **75.7 ms** to build and 73,859,072
bytes to write.

### The replication — 200 fresh questions, no search

Three things were held fixed, and each is an assert in `replicate.py` rather than a note:

- **the weights** — read out of `results.json` and never re-searched. Re-searching on the
  new set would turn an out-of-sample test back into an in-sample one, which is the whole
  thing this run exists to avoid;
- **the comparator** — `math-adaanchor`, the best single adapter *on the 50*. If the 200
  would have promoted a different adapter, it is not switched to. Switching would give the
  baseline the hindsight the blend was denied;
- **the items** — 200 new instances from the same ten generator families, written by a seed
  far from the scored and held-out seeds, and **proven disjoint** from all 70 previously
  used prompts before the run started.

| | configuration | accuracy | correct | seconds |
|---|---|---|---|---|
| **(c)** | **blend** | **0.380** | 76/200 | 3562 |
| **(a)** | **base model** | **0.290** | 58/200 | 1995 |
| (b) | `math-adaanchor` | 0.215 | 43/200 | 2910 |

| comparison | delta | exact McNemar | bootstrap CI | discordant |
|---|---|---|---|---|
| (c) − (a) blend vs base | **+9.0 pts** | p = 0.015 | [+0.020, +0.160] | 50 (34 blend-only, 16 base-only) |
| (c) − (b) blend vs single | **+16.5 pts** | p < 0.001 | [+0.100, +0.230] | 49 (41 blend-only, 8 single-only) |

**Verdict: PROCEED.** Both deltas clear the 5-point rule and both paired intervals exclude
zero, so the tripwire does not fire. This is the first run here where the point estimate and
the paired evidence agree.

The artifact was rebuilt from the frozen weights and re-verified before a question was
scored — applying the written file and applying the in-memory matrices gave a maximum logit
difference of `0.0` across 196 modules. Build time **117 ms**, same 73.9 MB.

### What moved between the two runs, and why it matters

| | on the 50 | on the 200 | change |
|---|---|---|---|
| base model | 0.300 | 0.290 | −1.0 pt |
| blend | 0.420 | 0.380 | −4.0 pts |
| `math-adaanchor` | 0.340 | 0.215 | **−12.5 pts** |

The base model held. The blend slipped about four points, which is roughly what a fit on 20
items should be expected to lose out of sample. **The single adapter fell 12.5 points.**

That matters for reading the headline. (c) − (a) is +9.0 against a baseline that barely
moved, so it is mostly the blend improving on base. (c) − (b) is +16.5, and **most of that
gap is the comparator degrading rather than the blend gaining**. The comparison against the
base model is the one that is not carried by a baseline falling over.

Both statements are true and both are in the table above. The second is the weaker claim.

### What this does and does not say

**Does.** On this base model, on this question distribution, a LoraHub blend of three public
adapters — fitted only on 20 held-out items — beats the base model by 9 points and the best
single adapter by 16.5 on 200 items none of them had seen, with paired evidence that agrees.
A blend of 74 MB can be built in 117 ms on a laptop CPU, with no training run and no GPU.

**Does not.** Nothing here says the network gives better answers, and the wording stays
"**can build** specialists on demand" rather than "better answers". This is one base model,
one size, one quantisation, one machine, and one question distribution. The result is that
*this* blending procedure works on *these* questions.

**Two caveats that belong beside the numbers, not in a footnote.**

- **The 200 are new instances of the same ten families, not a second domain.** They test
  that the fitted weights generalise across instances of the families they were fitted near.
  They do not test that blending transfers across domains, and 200 items must not be read as
  though the sample size had bought that. The in-between-domain question the experiment
  opens with is answered only within the maths-word-problem space.
- **The comparator was frozen, so the +16.5 is against a baseline that was not allowed to
  improve.** If the 200 would promote a different single adapter, this run does not know and
  does not say. That keeps the comparison honest in the direction that costs the blend, but
  it does mean "beats the best single adapter" means *the adapter that was best on the 50*.

Two further limits carry over unchanged and are not resolved by the replication:

- **The control set was not run on the 200.** It is a maths-only replication, so unlike the
  50-question run it says nothing about whether the blend costs general ability.
- **The contamination probe has still not been run.** The GSM8K template-space confound
  described under Limitations stands unmeasured. It inflates the baselines and any blend
  inheriting a memorising adapter inherits it too.

The evidence is `results.json` and `replication.json`, each carrying a per-question record —
the raw completion, what the extractor read, and whether that was right — for every
configuration, so every number above traces back to one. Greedy decoding and a seeded search
mean the scores reproduce exactly; the timings do not.

---

## The Ollama spike: does Stage 2 have a floor at all?

Stage 2 ends with a node running `ollama create` on a fused adapter. If Ollama cannot fuse a
Qwen2.5 LoRA, Stage 2 is empty — so this was tested **first**, before spending hours on the
evaluation, not last.

Result: **it works, on the GGUF path.**

PEFT-directory and bare-`.safetensors` `ADAPTER` lines both fail, in four different ways
(`no Modelfile or safetensors files found`, then `open adapter_config.json: no such file
or directory` regardless of where that file was placed). Ollama documents safetensors
adapter support for Llama, Mistral and Gemma; Qwen is not among them. Converting with
llama.cpp's `convert_lora_to_gguf.py` and pointing `ADAPTER` at the resulting GGUF succeeds.

Three independent confirmations that the adapter is genuinely fused rather than quietly
ignored, which is the failure that would matter:

1. `ollama create` prints `success`, where the safetensors path errored.
2. The model manifest contains `application/vnd.ollama.image.adapter` at 79,816,896 bytes.
3. The same prompt produces visibly different output — the base model writes an essay, the
   fused model writes the terse style its adapter was trained into, with a different final
   answer.

Conversion needs the base's `tokenizer.json`, `tokenizer_config.json`, `vocab.json` and
`merges.txt` (~18 MB) but not its 3 GB of weights, and the converter needs `sentencepiece`.

**Conclusion: Stage 2 is not empty.** Whether it is *worth building* is Stage 1's question.

---

## Limitations

Stated plainly, because several of them cut against the result rather than for it.

- **The maths set is in the GSM8K template space.** Fresh numbers do not remove template
  memorisation: a model tuned on ten thousand GSM8K unit-price problems has seen this shape
  of question whatever the figures. `probe.py` scores the base, the maths adapters and the
  blend on GSM8K's held-out test split and on MMLU high-school mathematics, and reports them
  beside the authored-set scores. A large GSM8K-versus-authored gap is the fingerprint of
  distribution fit rather than transfer. The confound cuts both ways — it inflates (b), the
  baseline, which is conservative, but a blend inheriting a memorising adapter inherits the
  inflation too, which is not.
- **20 held-out items is few.** Weight search on 20 items can fit noise, which biases
  against the blend. This is the conservative direction and is stated rather than hidden.
- **The control set is run for three configurations, not all seven.** It exists to detect
  whether the treatment cost general ability; seven would triple the cost to answer a
  question nobody asked.
- **One base model, one size, one quantisation, one machine.** 1.5B parameters is small.
  Nothing here generalises to 7B without being measured on 7B.
- **Adapter provenance is unknown.** Public adapters, unreviewed, with no statement of what
  they were trained on. One of them very likely saw GSM8K.
- **The verdict rule is crude on purpose.** Five points on 50 questions is two and a half
  questions. It is the threshold that was set, and it is reported alongside paired
  statistics rather than instead of them.

### How the verdict is decided

**(c) must beat both (a) and (b) by ≥5 points** on the 50 maths questions. That gates.

The paired statistics — exact McNemar on the discordant pairs, and a bootstrap CI on the
paired difference — are reported against every baseline and act as a **tripwire rather than
a veto**. At n=50 the CI half-width is roughly ±14 points, so a genuine 6-point gain will
essentially never reach significance; making the paired test a hard gate would throw away
real wins. But if the point estimate clears 5 points *while the CI still spans zero*, the
gain and the evidence disagree, and that is the case that stops the run and comes back to be
decided rather than resolving itself.

---

## Reproducing

```bash
cd experiments/lora-blend
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r requirements.txt

python make_questions.py         # writes the three JSON sets
python check_questions.py        # 1037 assertions over them
python run_eval.py --smoke       # 5 questions, hard gate: stop here if it fails
python run_eval.py               # the full run        -> results.json
python run_eval.py --resume      # the same, reusing whatever a prior run checkpointed

python make_replication_set.py   # the 200 fresh items -> questions_maths_200.json
python replicate.py --resume     # frozen-weight replication -> replication.json

python probe.py                  # contamination probe -> probe.json
```

`replicate.py` reads its weights out of `results.json` and **refuses to run** if that file
names a best single other than `math-adaanchor`, or if the frozen weights do not sum to 1 —
so the baseline cannot drift with the data without the run stopping. It is checkpointed per
*item* rather than per configuration, because at 200 items a configuration is roughly half
an hour and a per-config checkpoint would put all of it at risk on a kill.

`run_eval.py` loads the base model once (~3 GB, cached under `~/.cache/huggingface`) and
runs every configuration through it. `results.json` holds a per-question record for every
configuration — the raw completion, what the extractor read from it, and whether that was
right — so every number above traces back to one. Greedy decoding and a seeded search mean
re-running reproduces every score exactly; the `meta` timings are machine-dependent and do
not.

`--resume` exists because the run is long enough to be interrupted. `results.json` is
written only once, at the end, so a process killed during the scored phase discards every
configuration that had already finished — which happened, and cost an hour of search and
scoring with nothing on disk to show for it. The run now checkpoints each configuration as
it completes (under `blends/`, which is git-ignored, since it is build state and not
evidence) and `--resume` picks those up. The checkpoint records a signature of the seed,
budget and item counts that produced it and is discarded on any mismatch, so a resumed run
cannot silently splice together two different experiments. Without `--resume` the
checkpoint is deleted and the run is fresh.

### Files

| File | Purpose |
|---|---|
| `adapters.json` | the seven adapters, revisions pinned, blending precondition recorded |
| `make_questions.py` | authors all three sets, computing ground truth |
| `check_questions.py` | 1037 assertions proving the sets are internally sound |
| `harness.py` | loads the base once; adapters and blends swap in as a forward-pass delta |
| `blend.py` | LoraHub arithmetic, PEFT-layout artifact, gradient-free weight search |
| `run_eval.py` | every configuration × both sets → `results.json` |
| `make_replication_set.py` | authors the 200 fresh items, disjointness proven → `questions_maths_200.json` |
| `replicate.py` | frozen weights on the 200, no search, three configurations → `replication.json` |
| `probe.py` | GSM8K / MMLU contamination probe → `probe.json` |

### A note on `requirements.txt`

`datasets` (for `probe.py` only) resolves `huggingface-hub` to 2.x, which `transformers`
5.8.0 refuses — `huggingface-hub>=1.5.0,<2.0 is required`. The pin is in
`requirements.txt`; without it the probe's dependency silently breaks the scorer.
