"""NF4 block refuses to start without CUDA. Runs on the Mac; the CUDA path is untested here.

Run from the repository root:
    python3 -m unittest commonnet.tests.test_nf4_refusal -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from commonnet.backends.nf4_block import Nf4BlockRange  # noqa: E402
from commonnet.tests.test_split import make_checkpoint  # noqa: E402


class Nf4Refusal(unittest.TestCase):
    def test_refuses_without_cuda(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is present; the refusal path is not taken")
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "tiny")
            make_checkpoint(ckpt)
            with self.assertRaises(RuntimeError) as cm:
                Nf4BlockRange(ckpt, 0, 2)
            self.assertIn("no CUDA device", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


class Nf4Dtype(unittest.TestCase):
    """The factory hands the NF4 block bfloat16 compute unless told otherwise (float16 overflows to NaN on Qwen)."""

    def _dtype_for(self, env):
        import torch
        from unittest import mock
        from commonnet.net import factory
        seen = {}

        class Fake:
            def __init__(self, **kw):
                seen.update(kw)

        with mock.patch("commonnet.backends.nf4_block.Nf4BlockRange", Fake), \
                mock.patch.object(factory, "ServedBlock", lambda inner, *a, **k: inner):
            factory.build_block(0, 9, backend="nf4", checkpoint="/ckpt", environ=env)
        return seen["dtype"], torch

    def test_default_is_bfloat16_and_the_env_can_override(self):
        dtype, torch = self._dtype_for({})
        self.assertIs(dtype, torch.bfloat16)
        dtype, torch = self._dtype_for({"COMMONNET_DTYPE": "float16"})
        self.assertIs(dtype, torch.float16)
