"""Bounded checkpoint/replay for hybrid recurrent caches during speculative verification."""
import copy

CAPABILITY = "recurrent_rollback_v1"


def clone_mlx_caches(caches):
    import mlx.core as mx
    from mlx.utils import tree_map
    result = []
    for cache in caches:
        cloned = copy.copy(cache)
        # Attention appends beyond its saved offset; its committed prefix is
        # immutable. Only the fixed-size recurrent/conv arrays need copying.
        if not cache.is_trimmable():
            cloned.state = tree_map(lambda x: x + mx.zeros_like(x) if isinstance(x, mx.array) else x,
                                    cache.state)
            mx.eval(cloned.state)
        result.append(cloned)
    return result


def clone_session(session):
    saved = copy.copy(session)
    if hasattr(session, "caches"):
        saved.caches = clone_mlx_caches(session.caches)
    else:
        saved.cache = copy.copy(session.cache)
        # BlockRange appends KV tensors with torch.cat; share their committed
        # prefixes rather than duplicating a potentially long attention cache.
        saved.cache.keys = list(session.cache.keys)
        saved.cache.values = list(session.cache.values)
        saved.cache.layers = copy.deepcopy(session.cache.layers)
    return saved


class RecurrentRollback:
    def _save_recurrent(self, session_id, hidden, start_pos):
        if not (getattr(self, "qwen35", False) or getattr(self, "arch", "").startswith("qwen3_5")):
            return
        if not hasattr(self, "_replay"):
            self._replay = {}
        self._replay.pop(session_id, None)
        session = self._sessions.get(session_id)
        # Verification always has the confirmed last token plus >=1 draft.
        # Prefill and one-token plain decoding require no checkpoint.
        if start_pos and hidden.shape[1] > 1 and session is not None and session.length == start_pos:
            self._replay[session_id] = (clone_session(session), hidden.detach().clone(), start_pos)

    def _restore_recurrent(self, session_id, length):
        saved = getattr(self, "_replay", {}).get(session_id)
        if saved is None:
            raise ValueError("no verification checkpoint for recurrent rollback")
        session, hidden, start = saved
        if not start <= length <= start + hidden.shape[1]:
            raise ValueError("rollback exceeds the latest verification chunk")
        self._sessions[session_id] = clone_session(session)
        try:
            if length > start:
                self.forward(session_id, hidden[:, :length - start], start)
        finally:
            self._replay[session_id] = saved  # repeated/shorter rollback stays valid

    def _forget_recurrent(self, session_id):
        getattr(self, "_replay", {}).pop(session_id, None)
