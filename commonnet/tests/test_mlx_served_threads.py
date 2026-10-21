"""MLX blocks served over real HTTP: every request lands on a new server thread.

MLX streams are per-thread, so a block built on one thread and called from another
fails with "There is no Stream(gpu, N) in current thread". Needs Apple Metal.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

from commonnet.net.client import BlockClient  # noqa: E402
from commonnet.net.factory import build_block  # noqa: E402
from commonnet.net.server import BlockServer  # noqa: E402
from commonnet.net.transport import from_wire, to_wire  # noqa: E402
from commonnet.tests.qwen35_tiny import make_qwen35_checkpoint  # noqa: E402
from commonnet.tests.test_split import make_checkpoint  # noqa: E402

WIDTH = 64


@unittest.skipUnless(mx.metal.is_available(), "needs Metal")
class MlxServedAcrossThreads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="commonnet-mlx-thr-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def serve_and_run(self, ckpt, layers, **kw):
        if kw:  # tiny qwen3_5 needs group_size=32, which the env factory cannot pass
            from commonnet.backends import mlx_block
            from commonnet.net.factory import ServedBlock
            cls = getattr(mlx_block, "OnWorkerThread", None)
            made = cls(mlx_block.MlxBlockRange, checkpoint_dir=ckpt, start=0, end=layers, **kw) if cls \
                else mlx_block.MlxBlockRange(checkpoint_dir=ckpt, start=0, end=layers, **kw)
            block = ServedBlock(made, 0, layers, "metal")
        else:
            block = build_block(0, layers, backend="mlx", checkpoint=ckpt)
        srv = BlockServer(block).start()
        self.addCleanup(self._close, srv, block)
        gen = torch.Generator().manual_seed(1)
        threads = set()

        def call(sid, seq, pos):  # fresh client, fresh connection, fresh server thread
            h = torch.randn(1, seq, WIDTH, generator=gen).half()
            out = BlockClient(token=None, retries=0).forward(srv.url, sid, to_wire(h), pos)
            self.assertEqual(tuple(from_wire(out).shape), (1, seq, WIDTH))
            threads.add(threading.get_ident())

        c = BlockClient(token=None, retries=0)
        c.open(srv.url, "a")
        call("a", 5, 0)
        for i in range(3):
            call("a", 1, 5 + i)
        c.open(srv.url, "b")
        call("b", 4, 0)
        call("b", 1, 4)
        c.close(srv.url, "a")
        c.close(srv.url, "b")

    @staticmethod
    def _close(srv, block):
        srv.stop()
        block.shutdown()
        assert not [t for t in threading.enumerate() if t.name.startswith("mlx-worker") and t.is_alive()]

    def test_qwen3_prefill_decode_second_session(self):
        ckpt = os.path.join(self.tmp, "q3")
        make_checkpoint(ckpt)
        self.serve_and_run(ckpt, 4)

    def test_qwen35_hybrid_prefill_decode_second_session(self):
        ckpt = os.path.join(self.tmp, "q35")
        make_qwen35_checkpoint(ckpt, layers=8)
        self.serve_and_run(ckpt, 8, bits=None, group_size=32)


if __name__ == "__main__":
    unittest.main()
