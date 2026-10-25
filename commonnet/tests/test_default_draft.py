"""Default-on speculative decoding in the node chat API: when the drafter is used, when it silently is not."""
import os
import unittest
from unittest import mock

from commonnet import serve
from commonnet.tests.test_speculative import EPS, PROMPT, VOCAB, Rig, make_client
from commonnet.bench.synth import OracleDrafter


class FakeTransport:
    def __init__(self, caps=("chain_v1",)):
        self.caps = caps

    def capabilities(self, ep):
        return self.caps


class FakeNetwork:
    def __init__(self, transport):
        self.transport = transport

    def plan(self):
        return EPS, self.transport


class FakeClient:
    transport = None

    def generate_stream(self, endpoints, ids, n, **kw):
        yield from range(n)


def backend(transport=None, *, auto=True, no_spec=None, factory=None, env=None):
    with mock.patch.dict(os.environ, env or {}, clear=False):
        if env is None:
            os.environ.pop("COMMONNET_DRAFT", None)
        b = object.__new__(serve.NetworkBackend)
        b.network = FakeNetwork(transport or FakeTransport())
        b.client, b.drafter, b.draft_k, b.draft_note, b.stop_ids = FakeClient(), None, 4, "off", []
        b.auto_draft = auto and not no_spec and serve.draft_enabled()
        b._auto_failed = False
        b._drafter_factory = factory
        b.tokenizer = None
    return b


class Eligibility(unittest.TestCase):
    def drafter_for(self, b, temperature):
        eps, tr = b.network.plan()
        return b._lazy_drafter(eps, tr, temperature)

    def test_greedy_is_eligible_and_cached(self):
        calls = []
        b = backend(factory=lambda: calls.append(1) or object())
        d = self.drafter_for(b, 0)
        self.assertIsNotNone(d)
        b.drafter = d
        self.assertEqual(len(calls), 1)

    def test_sampled_request_never_loads_the_drafter(self):
        calls = []
        b = backend(factory=lambda: calls.append(1) or object())
        self.assertIsNone(self.drafter_for(b, 0.7))
        self.assertEqual(calls, [])

    def test_missing_capability_falls_back(self):
        b = backend(FakeTransport(caps=("ids_in",)), factory=lambda: object())
        self.assertIsNone(self.drafter_for(b, 0))

    def test_load_failure_falls_back_and_is_not_retried(self):
        calls = []
        def boom():
            calls.append(1)
            raise ImportError("no mlx_lm")
        b = backend(factory=boom)
        self.assertIsNone(self.drafter_for(b, 0))
        self.assertIsNone(self.drafter_for(b, 0))
        self.assertEqual(len(calls), 1)
        self.assertIn("off", b.draft_note)

    def test_opt_out_flag_and_env(self):
        self.assertFalse(backend(auto=False, factory=lambda: object()).auto_draft)
        b = backend(factory=lambda: object(), env={"COMMONNET_DRAFT": "0"})
        self.assertFalse(b.auto_draft)
        self.assertIsNone(self.drafter_for(b, 0))

    def test_unsupported_model_manifest(self):
        b = backend(no_spec="hybrid linear attention", factory=lambda: object())
        self.assertIsNone(self.drafter_for(b, 0))

    def test_shipped_manifests(self):
        import json, pathlib
        d = pathlib.Path(serve.__file__).parent / "networks"
        flags = {p.stem: "no_speculation" in json.loads(p.read_text()).get("model", {}) for p in d.glob("*.json")
                 if "model" in json.loads(p.read_text())}
        for name in ("bonsai-2-9b", "bonsai-2-27b"):
            self.assertTrue(flags[name], name)
        for name in ("qwen3-8b", "bonsai-8b"):
            if name in flags:
                self.assertFalse(flags[name], name)


class IdenticalOutput(unittest.TestCase):
    def test_backend_speculative_equals_plain_greedy(self):
        rig = Rig()
        self.addCleanup(rig.stop)
        tr = rig.transport()
        plain = list(make_client(tr).generate_stream(EPS, PROMPT, 30))
        truth = list(make_client(rig.transport()).generate_stream(EPS, PROMPT, 40))
        b = backend(tr, factory=lambda: OracleDrafter(len(PROMPT), truth, 0.7, vocab=VOCAB, seed=3))
        b.client = make_client(tr)
        b.network = FakeNetwork(tr)
        with mock.patch.object(type(tr), "capabilities", lambda self, ep: ("chain_v1",)):
            out = list(b._generate(PROMPT, 30, 0, 1.0, None))
        self.assertEqual(out, plain)
        self.assertIsNotNone(b.drafter)   # speculation really was on


if __name__ == "__main__":
    unittest.main()
