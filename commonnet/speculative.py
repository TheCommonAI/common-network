"""Speculative decoding for the layer network: a small local drafter proposes k tokens, the network checks them all in
one pass through the stages. On a token chain, the final peer verifies the longest
correct prefix and returns it plus a correction/bonus ID; the client rolls caches
back consistently. Legacy routes compare locally.

Each plain token costs one trip through every stage. A round here costs one trip too, but yields 1 + (accepted
drafts) tokens, so the per-hop latency is shared across them.

Greedy only: the output is exactly what plain greedy decoding gives (a draft is kept only if it equals the network's
own argmax at that position). temperature > 0 falls back to plain decoding.

    drafter = MlxDrafter("mlx-community/Qwen3-0.6B-4bit", target_tokenizer=tok)   # refuses a tokenizer mismatch
    for token in generate_stream(client, endpoints, prompt_ids, 256, drafter=drafter, k=8): ...

generate_stream() always works: with no drafter, an unsupported server on the route, or temperature > 0 it runs the
client's own plain decoding.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import torch

DEFAULT_K = 8
ADAPT_LOW = 0.3          # a round that accepted less than this share of its drafts shrinks the next draft
MIN_K = 1


class SpeculationRefused(RuntimeError):
    """The drafter cannot be used with this target (tokenizer mismatch, ...)."""


def check_tokenizers_match(target, drafter) -> None:
    """Refuse unless both tokenizers map exactly the same strings to the same ids (including special tokens)."""
    tv, dv = target.get_vocab(), drafter.get_vocab()
    if tv != dv:
        diff = sum(1 for t, i in tv.items() if dv.get(t) != i) + sum(1 for t in dv if t not in tv)
        raise SpeculationRefused(f"drafter tokenizer differs from the target's ({len(tv)} vs {len(dv)} tokens, "
                                 f"{diff} differ); refusing to speculate")
    ta = {t.content: i for i, t in getattr(target, "added_tokens_decoder", {}).items()}
    da = {t.content: i for i, t in getattr(drafter, "added_tokens_decoder", {}).items()}
    if ta != da:
        raise SpeculationRefused("drafter and target special tokens differ; refusing to speculate")


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    emitted: int = 0
    wall_s: float = 0.0
    k_used: list = field(default_factory=list)
    mode: str = ""          # "speculative" or "plain: <reason>"

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0


class MlxDrafter:
    """A small mlx_lm model on this machine, greedy, with its own KV cache that follows the accepted text."""

    def __init__(self, repo_or_path: str, *, target_tokenizer=None):
        from mlx_lm import load
        self.name = repo_or_path
        self.model, self.tokenizer = load(repo_or_path)
        if target_tokenizer is not None:
            check_tokenizers_match(target_tokenizer, self.tokenizer)
        self._cache = None
        self._toks: list[int] = []

    def reset(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache
        self._cache = make_prompt_cache(self.model)
        self._toks = []

    def propose(self, context: list[int], k: int) -> list[int]:
        import mlx.core as mx
        from mlx_lm.models.cache import trim_prompt_cache
        if self._cache is None:
            self.reset()
        common = 0
        for a, b in zip(self._toks, context):
            if a != b:
                break
            common += 1
        common = min(common, len(context) - 1)       # always feed at least the last context token
        if common < len(self._toks):
            trim_prompt_cache(self._cache, len(self._toks) - common)
            del self._toks[common:]
        feed = list(context[common:])
        out: list[int] = []
        for _ in range(k):
            logits = self.model(mx.array([feed]), cache=self._cache)
            nxt = int(mx.argmax(logits[0, -1]).item())
            self._toks.extend(feed)
            out.append(nxt)
            feed = [nxt]
        return out


def _truncate_all(client, endpoints, session_id: str, length: int) -> None:
    for ep in endpoints:
        client.transport.truncate(ep, session_id, length, defer=True)


def _verified_prefix(reply: dict, drafts: list[int], vocab_size: int) -> tuple[int, int]:
    """Validate the compact verification answer before using it to trim caches."""
    if not isinstance(reply, dict):
        raise ValueError("final peer returned a malformed verification answer")
    accepted, next_id, ids = reply.get("accepted"), reply.get("next"), reply.get("ids")
    if (type(accepted) is not int or not 0 <= accepted <= len(drafts)
            or type(next_id) is not int or not 0 <= next_id < vocab_size
            or not isinstance(ids, list) or len(ids) != accepted + 1
            or any(type(t) is not int for t in ids)
            or ids[:accepted] != drafts[:accepted] or ids[-1] != next_id):
        raise ValueError("final peer returned a malformed verification answer")
    return accepted, next_id


def can_speculate(client, endpoints) -> tuple[bool, str]:
    t = client.transport
    if not hasattr(t, "supports_truncate") or not hasattr(t, "truncate"):
        return False, "transport cannot roll back a session"
    try:
        ok = t.supports_truncate(endpoints)
    except Exception as e:  # a failed probe is not worth failing the request for
        return False, f"probe failed: {type(e).__name__}"
    return (True, "") if ok else (False, "a server on the route does not support rollback (old version)")


@torch.inference_mode()
def speculative_stream(client, endpoints, prompt_ids, max_new_tokens: int, drafter, *, k: int = DEFAULT_K,
                       adaptive: bool = True, stop_ids=None, stats: SpecStats | None = None):
    """Greedy speculative decoding. Yields exactly the token ids plain greedy decoding would."""
    client.check_route(endpoints, client.config.num_hidden_layers)
    if not prompt_ids:
        raise ValueError("prompt is empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    stats = stats if stats is not None else SpecStats()
    stats.mode = "speculative"
    if max_new_tokens < 1:
        return
    stops = frozenset(stop_ids or ())
    session_id = uuid.uuid4().hex
    t_start = time.perf_counter()
    if hasattr(drafter, "reset"):
        drafter.reset()
    produced, cur_k = 0, k
    try:
        use_chain = client._use_chain(endpoints)
        use_ids = use_chain or client._use_ids(endpoints)
        if use_ids:
            last = int(client._step_ids(endpoints, session_id, list(prompt_ids), 0, "last", chain=use_chain)[-1])
        else:
            ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=client.device)
            hidden = client._run_blocks(endpoints, session_id, client.embed(ids).to(client.dtype), 0)
            last = int(client._logits(hidden[:, -1:])[0, -1].argmax())
        pos = len(prompt_ids)
        context = list(prompt_ids)
        while True:
            if last in stops:
                return
            produced += 1
            yield last
            stats.emitted += 1
            if produced == max_new_tokens:
                return
            context.append(last)
            want = min(cur_k, max_new_tokens - produced)       # never draft past the end
            drafts = [int(d) for d in drafter.propose(context, want)][:want]
            if use_chain:
                reply = client.transport.chain_step(endpoints, session_id, [last] + drafts, pos,
                                                    which="all", drafts=drafts)
                m, next_id = _verified_prefix(reply, drafts, client.config.vocab_size)
            else:
                if use_ids:
                    greedy = client._step_ids(endpoints, session_id, [last] + drafts, pos, "all")
                else:
                    feed = torch.tensor([[last] + drafts], dtype=torch.long, device=client.device)
                    hidden = client._run_blocks(endpoints, session_id, client.embed(feed).to(client.dtype), pos)
                    greedy = client._logits(hidden)[0].argmax(dim=-1).tolist()
                m = 0
                while m < len(drafts) and drafts[m] == greedy[m]:
                    m += 1
                next_id = greedy[m]
            stats.rounds += 1
            stats.drafted += len(drafts)
            stats.accepted += m
            stats.k_used.append(len(drafts))
            if m < len(drafts):
                _truncate_all(client, endpoints, session_id, pos + 1 + m)
            pos += 1 + m
            if adaptive:
                if drafts and m / len(drafts) < ADAPT_LOW:
                    cur_k = max(MIN_K, cur_k // 2)
                elif m == len(drafts):
                    cur_k = min(k, cur_k * 2 if cur_k < k else k)
            for d in drafts[:m]:
                if d in stops:
                    return
                produced += 1
                yield d
                stats.emitted += 1
                context.append(d)
                if produced == max_new_tokens:
                    return
            last = next_id
    finally:
        stats.wall_s += time.perf_counter() - t_start
        for ep in endpoints:
            client.transport.close(ep, session_id)


def generate_stream(client, endpoints, prompt_ids, max_new_tokens: int, *, drafter=None, k: int = DEFAULT_K,
                    adaptive: bool = True, temperature: float = 0.0, stats: SpecStats | None = None, **plain):
    """Speculative when it can be, plain decoding (the client's own generate_stream) when it cannot."""
    stats = stats if stats is not None else SpecStats()
    reason = None
    if drafter is None:
        reason = "no drafter"
    elif temperature and temperature > 0:
        reason = "temperature > 0 (greedy only)"
    else:
        ok, why = can_speculate(client, endpoints)
        if not ok:
            reason = why
    if reason is not None:
        stats.mode = f"plain: {reason}"
        return client.generate_stream(endpoints, prompt_ids, max_new_tokens, temperature=temperature, **plain)
    return speculative_stream(client, endpoints, prompt_ids, max_new_tokens, drafter, k=k, adaptive=adaptive,
                              stop_ids=plain.get("stop_ids"), stats=stats)
