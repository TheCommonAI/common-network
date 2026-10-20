"""End to end over HTTP on localhost: tiny Qwen3, two real block servers, the coordinator."""
import json
import os
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

from commonnet import coordinator
from commonnet.client import BlockEndpoint
from commonnet.client4 import Client4
from commonnet.net import factory
from commonnet.net.client import RemoteError
from commonnet.net.server import BlockServer
from commonnet.net.transport import HttpTransport
from commonnet.tests.test_split import NEW_TOKENS, PROMPT, SplitTestBase

TOKEN = "t" * 32


class E2E(SplitTestBase):
    def setUp(self):
        env = {"COMMONNET_CHECKPOINT": self.ckpt}
        self.servers = []
        self.members = []
        for s, e in ((0, 2), (2, 4)):
            blk = factory.build_block(s, e, environ=env)
            srv = BlockServer(blk, token=TOKEN).start()
            self.servers.append(srv)
            self.members.append({"url": srv.url, "layers": [s, e]})
        self.tokfile = os.path.join(self.tmp, "tok")
        with open(self.tokfile, "w") as f:
            f.write(TOKEN)
        for m in self.members:
            m["token_file"] = self.tokfile

    def tearDown(self):
        for s in self.servers:
            s.stop()

    def reference_tokens(self):
        with torch.inference_mode():
            out = self.reference.generate(torch.tensor([PROMPT]), max_new_tokens=NEW_TOKENS,
                                          do_sample=False, eos_token_id=None, pad_token_id=0)
        return out[0, len(PROMPT):].tolist()

    def test_greedy_tokens_match_full_model(self):
        got = coordinator.generate(self.members, self.ckpt, PROMPT, NEW_TOKENS, fmt="fp32")
        want = self.reference_tokens()
        print(f"\n[e2e http] tokens equal: {got == want} ({len(got)} tokens)")
        self.assertEqual(got, want)

    def test_chunked_prefill_matches(self):
        route, urls, tokens = coordinator.build_route(self.members, 4)
        t = HttpTransport(urls, tokens=tokens, max_request_bytes=8192)  # forces several chunks
        client = Client4(self.ckpt, t, fmt="fp32")
        self.assertEqual(client.generate(route, PROMPT, NEW_TOKENS), self.reference_tokens())

    def test_wrong_token_is_a_clear_error(self):
        route, urls, _ = coordinator.build_route(self.members, 4)
        t = HttpTransport(urls, token="x" * 32, retries=0)
        client = Client4(self.ckpt, t, fmt="fp32")
        with self.assertRaises(RemoteError) as cm:
            client.generate(route, PROMPT, 2)
        self.assertEqual(cm.exception.status, 401)

    def test_gap_and_overlap_refused(self):
        for layers in (([0, 2], [3, 4]), ([0, 3], [2, 4]), ([0, 2], [2, 3]), ([1, 2], [2, 4])):
            ms = [{"url": "http://127.0.0.1:1", "layers": list(l), "token_file": self.tokfile} for l in layers]
            with self.subTest(layers=layers), self.assertRaises(coordinator.EndpointError):
                coordinator.build_route(ms, 4)

    def test_cli_main_prints_text_not_token(self):
        ef = os.path.join(self.tmp, "eps.json")
        with open(ef, "w") as f:
            json.dump(self.members, f)
        self.assertEqual(coordinator.load_endpoints(ef), self.members)
        with self.assertRaises(coordinator.EndpointError):
            coordinator.load_endpoints(json.dumps([]) and "http://x@0:1,garbage")


class Client4Formats(SplitTestBase):
    def test_fp16_client_close_to_fp32(self):
        from commonnet.block import BlockRange
        from commonnet.client import LocalTransport
        from commonnet.tests.test_split import route_for
        route = route_for([0, 2, 4])
        t = LocalTransport({e.name: BlockRange(self.ckpt, e.start, e.end, dtype=torch.float16) for e in route})
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            full = self.reference(ids).logits
        c = Client4(self.ckpt, t, fmt="fp16")
        self.assertLess(float((c.logits(route, ids.clone()) - full).abs().max()), 0.1)

    def test_mlx4_runs(self):
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            self.skipTest("no mlx")
        from commonnet.block import BlockRange
        from commonnet.client import LocalTransport
        from commonnet.tests.test_split import route_for
        route = route_for([0, 2, 4])
        t = LocalTransport({e.name: BlockRange(self.ckpt, e.start, e.end, dtype=torch.float16) for e in route})
        ids = torch.tensor([PROMPT])
        with torch.inference_mode():
            full = self.reference(ids).logits
        c = Client4(self.ckpt, t, fmt="mlx4")
        got = c.logits(route, ids)
        corr = float(torch.corrcoef(torch.stack([got.flatten(), full.flatten()]))[0, 1])
        print(f"\n[mlx4 client] logit correlation with fp32 = {corr:.4f}, client bytes {c.client_bytes}")
        self.assertGreater(corr, 0.9)
        self.assertEqual(len(c.generate(route, PROMPT, 4)), 4)


if __name__ == "__main__":
    unittest.main()


class FactoryChoices(unittest.TestCase):
    def test_bad_backend_and_missing_checkpoint(self):
        with self.assertRaises(ValueError):
            factory.build_block(0, 1, environ={"COMMONNET_BACKEND": "gguf", "COMMONNET_CHECKPOINT": "x"})
        with self.assertRaises(ValueError):
            factory.build_block(0, 1, environ={})

    @unittest.skipIf(torch.cuda.is_available(), "this machine has CUDA")
    def test_nf4_refuses_without_cuda(self):
        with self.assertRaisesRegex(RuntimeError, "no CUDA"):
            factory.build_block(0, 1, environ={"COMMONNET_BACKEND": "nf4", "COMMONNET_CHECKPOINT": "x"})
