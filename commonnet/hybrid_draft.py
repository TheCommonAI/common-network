"""Qwen3.5 drafters restore the committed context cache instead of trimming recurrence."""
import copy

import torch

from commonnet.speculative import MlxDrafter, TorchDrafter


class HybridMlxDrafter(MlxDrafter):
    def reset(self):
        super().reset()
        self._committed = None

    def propose(self, context, k):
        import mlx.core as mx
        from commonnet.recurrent import clone_mlx_caches
        saved = getattr(self, "_committed", None)
        if saved is not None and context[:len(saved[0])] == saved[0]:
            prefix, cache = saved
            self._cache = clone_mlx_caches(cache)
            feed = context[len(prefix):]
        else:
            self.reset()
            feed = list(context)
        # A new verification round always adds at least its correction/bonus.
        if not feed:
            self.reset()
            feed = list(context)
        out = []
        for i in range(k):
            logits = self.model(mx.array([feed]), cache=self._cache)
            nxt = int(mx.argmax(logits[0, -1]).item())
            if i == 0:
                self._committed = (list(context), clone_mlx_caches(self._cache))
            out.append(nxt)
            feed = [nxt]
        return out


class HybridTorchDrafter(TorchDrafter):
    def __init__(self, repo_or_path, *, target_tokenizer=None, device=None, model=None, tokenizer=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            import os
            from transformers import AutoModelForImageTextToText, AutoTokenizer
            kw = {}
            if os.environ.get("COMMONNET_CACHE"):
                kw["cache_dir"] = os.path.join(os.environ["COMMONNET_CACHE"], "hf")
            tokenizer = AutoTokenizer.from_pretrained(repo_or_path, **kw)
            model = AutoModelForImageTextToText.from_pretrained(
                repo_or_path, dtype=torch.bfloat16 if device == "cuda" else torch.float32, **kw)
        super().__init__(repo_or_path, target_tokenizer=target_tokenizer, device=device,
                         model=model, tokenizer=tokenizer)
        self._committed = None

    def reset(self):
        self._cache = None
        self._committed = None

    @torch.inference_mode()
    def propose(self, context, k):
        saved = self._committed
        if saved is not None and context[:len(saved[0])] == saved[0] and len(context) > len(saved[0]):
            prefix, cache = saved
            self._cache = copy.deepcopy(cache)
            feed = context[len(prefix):]
        else:
            self.reset()
            feed = list(context)
        out = []
        for i in range(k):
            result = self.model(input_ids=torch.tensor([feed], device=self.device),
                                past_key_values=self._cache, use_cache=True)
            self._cache = result.past_key_values
            nxt = int(result.logits[0, -1].argmax().item())
            if i == 0:
                self._committed = (list(context), copy.deepcopy(self._cache))
            out.append(nxt)
            feed = [nxt]
        return out
