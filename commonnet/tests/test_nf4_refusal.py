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
