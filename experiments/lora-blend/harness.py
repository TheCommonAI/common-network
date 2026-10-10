#!/usr/bin/env python3
"""One loaded base model, with adapters and blends swapped into it.

Design, and why it is this way
------------------------------

Applying a LoRA is `W_eff = W + (alpha/r) * B @ A`. Rather than merge that into the
weights, this wraps each of the seven target projections in a `LoRALinear` that adds the
delta during the forward pass. Swapping configuration is then seven pointer assignments
per layer, the base weights are never touched, and returning to the base model is
`enabled = False` rather than a reload.

That matters here for a reason beyond speed. Merging would mean a full copy of the model
per configuration (roughly 3 GB in fp16), float accumulation into the base weights whose
order depends on which configurations ran before, and an eval that cannot be re-run
identically without reloading from disk. The wrapper removes all three problems: the
blend matrices for the whole model are ~12 MB, and config N+1 cannot be contaminated by
config N.

The same mechanism serves an adapter and a blend, because a LoraHub blend is the identical
expression with `A` and `B` replaced by the weighted sums across adapters. A blend is not
a different kind of object here; it is a different set of matrices. That is the whole
reason `blend.py` can be short.

Adapter matrices are held in float32 regardless of the base model's dtype. In fp16 the
product of two summed matrices loses precisely the small differences between candidate
blends that the weight search is trying to measure.

Nothing here imports `peft`. See README.md, "Why there is no PEFT here".
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent

# Anything ending in one of these leaf names, and being an nn.Linear, is a LoRA target.
# Read from adapters.json rather than hardcoded so that the harness and the registry
# cannot drift apart -- if an adapter with a different target set is ever added, the
# assertion in `load_registry` fails loudly here instead of silently applying nothing.
DEFAULT_TARGET_LEAVES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


# --- answer extraction ---------------------------------------------------------------
#
# One extractor, applied identically to every configuration. A parser tuned per config
# would measure the parser as much as the model, so the rule is fixed in advance and
# never revisited after seeing which config it favours. It is crude on purpose: "the
# last number in the completion wins" is defensible, and being uniformly crude is better
# than being cleverly uneven.

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_number(text: str) -> float | None:
    """The last number appearing anywhere in the completion, or None if there is none."""
    found = _NUMBER.findall(text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:  # pragma: no cover - regex makes this unreachable
        return None


def number_correct(got: float | None, expected: str, places: int) -> bool:
    """Whether `got` is `expected` rounded to the item's own stated precision.

    The tolerance is half a unit in the last stated place, which is exactly "would this
    round to the right answer". For a places=0 item that is +-0.5, so 81 counts and 80.8
    does not. Deliberately not looser than that: a 1% relative tolerance would accept 299
    for 296.82, at which point the metric stops measuring arithmetic.
    """
    if got is None:
        return False
    tolerance = 0.5 * (10 ** -places) + 1e-9
    return abs(got - float(expected)) <= tolerance


def _normalise_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_correct(text: str, expected: str) -> bool:
    """Whether the expected answer appears in the completion as a whole word.

    Word boundaries matter: "voltage" must not satisfy the answer "volt". Being generous
    about *where* in the completion the answer appears is deliberate -- a model that
    writes a sentence around its answer has still answered -- and because the rule is the
    same for every config, the generosity cancels.
    """
    haystack = _normalise_text(text)
    needle = _normalise_text(expected)
    if not needle:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def score(completion: str, item: dict) -> tuple[object, bool]:
    """Extract and mark one completion. Returns (extracted, correct)."""
    if item["answer_type"] == "number":
        got = extract_number(completion)
        return got, number_correct(got, item["answer"], item["places"])
    return completion, text_correct(completion, item["answer"])


# --- device and dtype ---------------------------------------------------------------

def pick_device() -> tuple[str, torch.dtype]:
    """Prefer MPS, then CUDA, then CPU. Overridable for debugging.

    fp16 on MPS rather than bf16: Apple's Metal backend has a much longer history with
    fp16, and this is a 1.5B model where the extra mantissa bits of bf16 buy nothing that
    the float32 LoRA delta does not already provide.
    """
    forced_device = os.environ.get("LORA_BLEND_DEVICE")
    forced_dtype = os.environ.get("LORA_BLEND_DTYPE")
    if forced_device:
        table = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        return forced_device, table.get(forced_dtype or "", torch.float32)
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


# --- the wrapper --------------------------------------------------------------------

def weighted_factor_sums(pairs, weights) -> tuple:
    """`sum_i w_i A_i` and `sum_i w_i B_i` over a list of (A, B) pairs.

    This is the single implementation of the LoraHub sum, shared by
    `LoRALinear.set_blend` (which scores a configuration) and `blend.blend_sums` (which
    writes the artifact). It exists as one function because it previously existed as two,
    and the two copies drifted: the version in this file omitted the weight on its first
    term, so the blend the harness *scored* was not the blend that got *written*. Nothing
    in the harness could see that -- both paths ran to completion and both produced
    plausible numbers -- and it surfaced only when `blend.verify_artifact` compared the
    next-token distributions of the two. One implementation, used by both callers, is the
    fix rather than a corrected copy of the same formula.

    Every term is multiplied by its weight, including the first. That is the easy thing to
    get wrong here, because the `A is None` branch that seeds the accumulator reads like a
    plain assignment rather than a weighted one.

    Returns (None, None) when every weight is zero; callers treat that as `clear()`.
    """
    A = None
    B = None
    for weight, (a_i, b_i) in zip(weights, pairs):
        if weight == 0.0:
            continue
        A = weight * a_i if A is None else A + weight * a_i
        B = weight * b_i if B is None else B + weight * b_i
    return A, B


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a swappable LoRA delta.

    The delta is computed in float32 and cast back, so the wrapper behaves identically
    whether the base model is fp16, bf16 or fp32, and so a blend's summed matrices keep
    their precision. `A` and `B` are registered as buffers (initialised to None) rather
    than plain attributes so that `model.to(device)` moves them with everything else.
    """

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("A", None)
        self.register_buffer("B", None)
        self.scaling = 0.0
        self.enabled = False

    def set_lora(self, A: torch.Tensor, B: torch.Tensor, scaling: float) -> None:
        self.A = A.to(dtype=torch.float32, device=self.base.weight.device)
        self.B = B.to(dtype=torch.float32, device=self.base.weight.device)
        self.scaling = float(scaling)
        self.enabled = True

    def set_blend(self, pairs, weights, scaling: float) -> None:
        """Weighted sums of (A, B) across adapters, then the product.

        The sum is `weighted_factor_sums`, not written out here. See that function: this
        method used to hold its own copy of the formula and the copy was wrong.
        """
        A, B = weighted_factor_sums(pairs, weights)
        if A is None:  # every weight was zero
            self.clear()
            return
        self.set_lora(A, B, scaling)

    def clear(self) -> None:
        self.enabled = False
        self.A = None
        self.B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            delta = (x.float() @ self.A.t()) @ self.B.t()
            out = out + (self.scaling * delta).to(out.dtype)
        return out

    def extra_repr(self) -> str:
        shape = "" if self.A is None else f", r={self.A.shape[0]}"
        return f"enabled={self.enabled}{shape}"


def wrap_targets(model: nn.Module, leaves=DEFAULT_TARGET_LEAVES) -> dict[str, LoRALinear]:
    """Replace every target projection with a LoRALinear wrapping it.

    Matches on the leaf name rather than on the full path, because the path differs
    between architectures and the leaf does not. Returns {module_path: wrapper}, where
    the paths are the same strings a PEFT adapter's tensor names reduce to, which is what
    lets `load_adapter_matrices` line the two up without a mapping table.
    """
    wrapped: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in leaves or not isinstance(module, nn.Linear):
            continue
        if name.count(".") < 2:  # a stray top-level linear, not a decoder projection
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        wrapper = LoRALinear(module)
        setattr(parent, leaf, wrapper)
        wrapped[name] = wrapper
    return wrapped


# --- the harness --------------------------------------------------------------------

class Harness:
    """A loaded base model plus the wrapped projections, ready to take any config."""

    def __init__(self, base_model: str, target_leaves=DEFAULT_TARGET_LEAVES,
                 device: str | None = None, dtype: torch.dtype | None = None) -> None:
        self.base_model = base_model
        self.target_leaves = tuple(target_leaves)
        self.device, self.dtype = (
            (device, dtype) if device and dtype else pick_device()
        )
        self.model = None
        self.tokenizer = None
        self.wrapped: dict[str, LoRALinear] = {}
        self.load_seconds = 0.0
        self._modules_by_leaf = {}

    def load(self) -> "Harness":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model, dtype=self.dtype)
        except TypeError:
            # transformers < 5 spells this torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.wrapped = wrap_targets(self.model, self.target_leaves)
        if not self.wrapped:
            raise RuntimeError(
                f"wrapped no modules for leaves {self.target_leaves} in {self.base_model}")
        self._modules_by_leaf = {n.rsplit(".", 1)[-1] for n in self.wrapped}
        self.load_seconds = time.perf_counter() - started
        return self

    # -- configuration ------------------------------------------------------------

    def clear(self) -> None:
        """Return to the untouched base model."""
        for wrapper in self.wrapped.values():
            wrapper.clear()

    def use_single(self, matrices: dict, scaling: float) -> int:
        """Apply one adapter's matrices. Returns how many modules were set.

        Clears first, and that line is load-bearing rather than tidy. An adapter need not
        target every projection -- the medical adapter covers q/k/v/o and not the MLP --
        so a caller that swapped from a 7-target adapter to a 4-target one without
        clearing would leave three projections still carrying the *previous* adapter's
        matrices, and score a hybrid while reporting a single adapter. Every `use_*`
        entry point clears for the same reason.
        """
        self.clear()
        applied = 0
        for path, wrapper in self.wrapped.items():
            pair = matrices.get(path)
            if pair is None:
                continue
            wrapper.set_lora(pair[0], pair[1], scaling)
            applied += 1
        return applied

    def use_blend(self, matrices_list: list[dict], weights: list[float], scaling: float) -> int:
        """Apply a LoraHub blend: sum of A and B across adapters, then the product.

        Note the shape of this. It is `(sum_i w_i A_i)` and `(sum_i w_i B_i)` that are
        formed first, and the product taken afterwards -- not a sum of per-adapter
        products. LoraHub's point is that the former is one rank-r adapter and the latter
        is a rank-(n*r) sum that no longer behaves like a low-rank update. Getting this
        backwards produces a plausible-looking model that is not a blend at all.
        """
        self.clear()
        applied = 0
        for path, wrapper in self.wrapped.items():
            pairs = [m[path] for m in matrices_list if path in m]
            if len(pairs) != len(matrices_list):
                continue  # an adapter lacking this module; skip rather than half-blend
            wrapper.set_blend(pairs, weights, scaling)
            applied += 1
        return applied

    def use_weights(self, matrices_list: list[dict], weights: list[float], scaling: float) -> int:
        """Single adapter if one weight is 1, blend otherwise. Keeps callers simple."""
        live = [(w, m) for w, m in zip(weights, matrices_list) if w != 0.0]
        if len(live) == 1 and live[0][0] == 1.0:
            return self.use_single(live[0][1], scaling)
        return self.use_blend([m for _, m in live], [w for w, _ in live], scaling)

    def describe_active(self) -> str:
        active = sum(1 for w in self.wrapped.values() if w.enabled)
        return f"{active}/{len(self.wrapped)} projections"

    # -- inference ----------------------------------------------------------------

    def build_prompt(self, question: str, answer_type: str) -> str:
        """The chat template plus a fixed instruction, identical across configurations.

        The instruction differs between the two sets only because what counts as an
        answer differs; within a set it is byte-identical for base, every adapter, the
        blend and the probe. No per-config prompt tuning, by construction.
        """
        instruction = (
            "Give your final answer as a single number."
            if answer_type == "number"
            else "Give your final answer as a short phrase."
        )
        messages = [{"role": "user", "content": f"{question}\n\n{instruction}"}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def next_token_logits(self, prompt: str) -> torch.Tensor:
        """Last-position logits for a raw prompt. Used to check two code paths agree."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.logits[0, -1].float().cpu()

    def generate(self, question: str, answer_type: str = "number",
                 max_new_tokens: int = 200) -> tuple[str, float]:
        """Greedy decode one answer. Returns (completion, seconds).

        Greedy because a sampled run would put a random variable inside the comparison
        between configurations, and the whole claim being tested is a difference of a
        few points. It also makes the run reproducible: same weights, same prompt, same
        tokens, every time.
        """
        prompt = self.build_prompt(question, answer_type)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


# --- adapter loading ----------------------------------------------------------------

def _split_lora_key(key: str) -> tuple[str | None, str | None]:
    """`base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` -> (path, 'A').

    Anchored on the last `model.layers.` occurrence and indifferent to everything before
    it, so the PEFT prefix variants (`base_model.model.`, a bare `model.`, a
    `.default.` adapter name) all reduce to the same path -- which is also what
    `named_modules()` yields, so no mapping table is needed.
    """
    tag = None
    for candidate in ("lora_A", "lora_B"):
        if f".{candidate}." in key:
            tag = candidate[-1]
            stem = key.split(f".{candidate}.")[0]
            break
    if tag is None:
        return None, None
    marker = "model.layers."
    position = stem.find(marker)
    if position < 0:
        return None, None
    return stem[position:], tag


def load_adapter_matrices(entry: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Download one adapter's weights and return {module_path: (A, B)} in float32."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        entry["hf_repo"], entry["weight_file"], revision=entry["revision"])
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        raw = load_file(path)
    else:
        # The one adapter shipping a pickle. `weights_only=True` refuses anything that
        # is not a plain tensor container, which is the behaviour we want on a file
        # pulled from the internet; the caller drops the adapter if this raises.
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise ValueError("adapter pickle is not a state dict")
        raw = raw.get("state_dict", raw)

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        if not torch.is_tensor(tensor):
            continue
        stem, tag = _split_lora_key(key)
        if stem is None:
            continue
        grouped.setdefault(stem, {})[tag] = tensor.to(torch.float32)

    matrices = {stem: (d["A"], d["B"]) for stem, d in grouped.items() if "A" in d and "B" in d}
    if not matrices:
        raise ValueError("no lora_A/lora_B pairs found in the adapter")
    return matrices


def load_registry(path: Path | None = None) -> dict:
    """Read adapters.json and assert the blending precondition actually holds.

    The assertion is the point. LoraHub-style averaging requires every adapter in a blend
    to agree on r, lora_alpha and target_modules, and a registry that merely *claims*
    they agree is worth nothing -- the failure mode is a blend that loads and produces
    slightly wrong numbers. So the claim is checked here, at load, every run.
    """
    registry = json.loads((path or HERE / "adapters.json").read_text())
    blendable = [a for a in registry["adapters"] if a.get("blendable")]
    if blendable:
        shapes = {(a["r"], a["lora_alpha"], tuple(a["target_modules"])) for a in blendable}
        if len(shapes) != 1:
            raise ValueError(f"blendable adapters disagree on (r, alpha, targets): {shapes}")
    return registry


def scaling_for(entry: dict) -> float:
    return entry["lora_alpha"] / entry["r"]


if __name__ == "__main__":
    # Minimal load-and-generate check: does the model come up, on what device, and how
    # fast? Run this before run_eval.py --smoke if the model is not already cached, since
    # the first invocation pays the download.
    import sys

    registry = load_registry()
    harness = Harness(registry["base_model"]).load()
    print(f"device       {harness.device} / {harness.dtype}")
    print(f"loaded in    {harness.load_seconds:.1f}s")
    print(f"wrapped      {harness.describe_active()}")

    question = ("A shop sells mugs at £0.50 each. Omar buys 5 of them and pays with a "
                "£20 note. How much change, in pounds, does Omar receive?")
    item = {"answer_type": "number", "answer": "17.50", "places": 2}

    completion, seconds = harness.generate(question)
    print(f"\nbase  ({seconds:.1f}s): {completion.strip()[:160]!r}")
    print(f"      -> {score(completion, item)}")

    adapters = [a for a in registry["adapters"] if a.get("blendable")]
    if adapters:
        entry = adapters[0]
        matrices = load_adapter_matrices(entry)
        applied = harness.use_single(matrices, scaling_for(entry))
        completion, seconds = harness.generate(question)
        print(f"\n{entry['id']} ({applied} modules, {seconds:.1f}s): "
              f"{completion.strip()[:160]!r}")
        print(f"      -> {score(completion, item)}")

    harness.clear()
    completion, _ = harness.generate("What is the capital of Hungary?")
    print(f"\nbase again: {completion.strip()[:160]!r}")
    sys.exit(0)
