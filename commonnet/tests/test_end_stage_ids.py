"""End-window placement and real end-table numerics on offline tiny checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

from commonnet.client import Client
from commonnet.mesh.manifest import Manifest, ManifestError
from commonnet.net.endio import EndIO, MissingTensors, caps_for
from commonnet.net.factory import build_block
from commonnet.tests.test_split import make_checkpoint


def metal_available():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except ImportError:
        return False


class EndStageTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="commonnet-ends-")
        cls.path = str(Path(cls.tmp.name) / "plain")
        make_checkpoint(cls.path)
        cls.tied = str(Path(cls.tmp.name) / "tied")
        make_checkpoint(cls.tied, tie=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_only_end_windows_advertise_and_load_their_tables(self):
        self.assertEqual(caps_for(self.path, 0, 1), ("ids_in",))
        self.assertEqual(caps_for(self.path, 1, 3), ())
        self.assertEqual(caps_for(self.path, 3, 4), ("head_out",))
        first = build_block(0, 1, backend="torch", checkpoint=self.path, environ={"COMMONNET_DTYPE": "float32"})
        middle = build_block(1, 3, backend="torch", checkpoint=self.path, environ={"COMMONNET_DTYPE": "float32"})
        last = build_block(3, 4, backend="torch", checkpoint=self.path, environ={"COMMONNET_DTYPE": "float32"})
        self.assertIsNone(middle.io)
        self.assertFalse(first.io._ready)
        self.assertFalse(last.io._ready)
        first.io.embed([1, 4])
        last.io.head_ids(torch.ones(1, 2, 64), "last", None, 0)
        self.assertEqual(first.io.loaded_names, ["model.embed_tokens.weight"])
        self.assertEqual(set(last.io.loaded_names), {"model.norm.weight", "lm_head.weight"})
        self.assertIsNone(first.io.head_mod)
        self.assertIsNone(last.io.embed_mod)

    def test_fp32_embedding_and_head_match_client_exactly(self):
        client = Client(self.path, None)
        io = EndIO(self.path, "torch", want_embed=True, want_head=True, dtype=torch.float32)
        ids = torch.tensor([[5, 17, 301, 42]])
        torch.testing.assert_close(io.embed(ids[0].tolist()), client.embed(ids), rtol=0, atol=0)
        hidden = torch.randn(1, 6, 64, generator=torch.Generator().manual_seed(64))
        expected = client._logits(hidden).argmax(-1)[0].tolist()
        self.assertEqual(io.head_ids(hidden, "all", None, 7), expected)
        self.assertEqual(io.head_ids(hidden, "last", None, 7), expected[-1:])
        self.assertEqual(io.nbytes, (512 * 64 * 2 + 64) * 4)

    def test_cpu_dtype_follows_served_decoder(self):
        block = build_block(0, 4, backend="cpu", checkpoint=self.path, environ={"COMMONNET_DTYPE": "float32"})
        block.io.embed([1, 2])
        self.assertEqual(block.io.dtype, block.inner.dtype)
        self.assertEqual(block.io.head_mod.weight.dtype, torch.float32)
        self.assertEqual(block.io.norm.weight.device, block.inner.device)
        self.assertEqual(block.io.head_ids(torch.ones(1, 2, 64), "last", None, 0),
                         Client(self.path, None)._logits(torch.ones(1, 2, 64)).argmax(-1)[0, -1:].tolist())

    def test_tied_head_is_supported_and_shared_on_one_device(self):
        io = EndIO(self.tied, "torch", want_embed=True, want_head=True)
        client = Client(self.tied, None)
        self.assertEqual(caps_for(self.tied, 0, 4), ("ids_in", "head_out"))
        hidden = io.embed([2, 5, 11])
        self.assertEqual(io.head_ids(hidden, "all", None, 0), client._logits(hidden).argmax(-1)[0].tolist())
        self.assertEqual(io.embed_mod.weight.data_ptr(), io.head_mod.weight.data_ptr())
        self.assertEqual(io.nbytes, (512 * 64 + 64) * 4)

    def test_bad_token_ids_are_rejected_before_loading_or_gpu_dispatch(self):
        io = EndIO(self.path, "torch", want_embed=True, want_head=False)
        for ids in ([], [-1], [512], [True], [1.5], [2**32 - 1]):
            with self.subTest(ids=ids), mock.patch.object(io, "_load") as load:
                with self.assertRaises(ValueError):
                    io.embed(ids)
                load.assert_not_called()

    def test_sampling_retries_are_position_deterministic(self):
        io = EndIO(self.path, "torch", want_embed=False, want_head=True)
        hidden = torch.randn(1, 5, 64, generator=torch.Generator().manual_seed(12))
        args = (hidden, "all", {"seed": 91, "temperature": 0.8, "top_p": .9}, 6)
        self.assertEqual(io.head_ids(*args), io.head_ids(*args))
        all_ids = io.head_ids(*args)
        self.assertEqual(io.head_ids(hidden[:, -1:], "last", args[2], 10), all_ids[-1:])

    def test_wrong_head_width_is_rejected_before_device_dispatch(self):
        io = EndIO(self.path, "torch", want_embed=False, want_head=True)
        with mock.patch.object(io, "_load") as load, self.assertRaises(ValueError):
            io.head_ids(torch.ones(1, 2, 63), "last", None, 0)
        load.assert_not_called()

    def test_ids_can_be_disabled_without_allocating_end_tables(self):
        block = build_block(0, 4, backend="torch", checkpoint=self.path, environ={"COMMONNET_IDS": "0"})
        self.assertEqual(block.caps, ())
        self.assertIsNone(block.io)

    def test_incomplete_or_missing_weight_shards_are_not_advertised(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(Path(self.path, "config.json").read_text())
            Path(d, "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "model.embed_tokens.weight": "missing.safetensors", "model.norm.weight": "missing.safetensors"}}))
            self.assertEqual(caps_for(d, 0, 4), ())
            Path(d, "model.safetensors.index.json").unlink()
            tensors = load_file(str(Path(self.path, "model.safetensors")))
            tensors.pop("lm_head.weight")
            tensors["lm_head.scales"] = torch.ones(512, 1)
            save_file(tensors, str(Path(d, "model.safetensors")))
            self.assertNotIn("head_out", caps_for(d, 0, 4))
            with self.assertRaises(MissingTensors):
                EndIO(d, "torch", want_embed=False, want_head=True)._load()


@unittest.skipUnless(metal_available(), "requires Metal to create an MLX quantized table")
class PackedEndTables(unittest.TestCase):
    def test_affine_packed_cpu_and_mlx_ends_match_reference(self):
        import mlx.core as mx
        from commonnet.client4 import Client4
        with tempfile.TemporaryDirectory() as d:
            plain, packed = Path(d, "plain"), Path(d, "packed")
            make_checkpoint(str(plain)); packed.mkdir()
            raw = load_file(str(plain / "model.safetensors"))
            tables = {"model.norm.weight": mx.array(raw["model.norm.weight"].numpy())}
            for name in ("model.embed_tokens", "lm_head"):
                weight = mx.array(raw[name + ".weight"].numpy()).astype(mx.float16)
                w, sc, bi = mx.quantize(weight, group_size=64, bits=4)
                for suffix, value in (("weight", w), ("scales", sc), ("biases", bi)):
                    tables[name + "." + suffix] = value
            mx.save_safetensors(str(packed / "model.safetensors"), tables)
            config = json.loads((plain / "config.json").read_text())
            config["quantization"] = {"bits": 4, "group_size": 64}
            (packed / "config.json").write_text(json.dumps(config))
            reference = Client4(str(packed), None, fmt="mlx4")
            ids = torch.tensor([[5, 17, 301, 42]])
            hidden = torch.randn(1, 4, 64, generator=torch.Generator().manual_seed(10))
            expected_ids = reference._logits(hidden).argmax(-1)[0].tolist()
            for backend in ("mlx", "cpu"):
                with self.subTest(backend=backend):
                    io = EndIO(str(packed), backend, want_embed=True, want_head=True, dtype=torch.float32)
                    torch.testing.assert_close(io.embed(ids[0].tolist()), reference.embed(ids), atol=5e-4, rtol=5e-4)
                    self.assertEqual(io.head_ids(hidden, "all", None, 0), expected_ids)

    def test_hadamard_pack_end_tables_match_effective_float_reference(self):
        from commonnet.tests.test_hadamard_pack import build, EFFECTIVE
        with tempfile.TemporaryDirectory() as d:
            packed, reference = build(d)
            client = Client(reference, None)
            io = EndIO(packed, "cpu", want_embed=True, want_head=True, dtype=torch.float32)
            ids = torch.tensor([[5, 17, 301, 42]])
            effective_embed = torch.from_numpy(EFFECTIVE["model.embed_tokens"])
            effective_head = torch.from_numpy(EFFECTIVE["lm_head"])
            torch.testing.assert_close(io.embed(ids[0].tolist()), torch.nn.functional.embedding(ids, effective_embed),
                                       atol=5e-5, rtol=5e-5)
            hidden = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(12))
            expected = torch.nn.functional.linear(client.norm(hidden), effective_head).argmax(-1)[0].tolist()
            self.assertEqual(io.head_ids(hidden, "all", None, 0), expected)
            self.assertEqual(caps_for(packed, 0, 4, "cpu"), ("ids_in", "head_out"))

    def test_mixed_plain_embedding_and_packed_head_are_detected_per_table(self):
        import mlx.core as mx
        with tempfile.TemporaryDirectory() as d:
            make_checkpoint(d)
            raw = load_file(str(Path(d, "model.safetensors")))
            tables = {n: mx.array(t.numpy()) for n, t in raw.items()}
            w, sc, bi = mx.quantize(tables["lm_head.weight"].astype(mx.float16), group_size=64, bits=4)
            tables.update({"lm_head.weight": w, "lm_head.scales": sc, "lm_head.biases": bi})
            mx.save_safetensors(str(Path(d, "model.safetensors")), tables)
            config = json.loads(Path(d, "config.json").read_text())
            config["quantization"] = {"bits": 4, "group_size": 64}
            Path(d, "config.json").write_text(json.dumps(config))
            io = EndIO(d, "mlx", want_embed=True, want_head=True)
            embedded = io.embed([5, 17])
            self.assertEqual(embedded.shape, (1, 2, 64))
            self.assertEqual(len(io.head_ids(embedded, "all", None, 0)), 2)


class EndMemoryPlacement(unittest.TestCase):
    def manifest(self, io_gib):
        return Manifest.from_dict({"id": "end-budgets", "model": {"num_layers": 4}, "boundaries": [0, 1, 3, 4],
                                   "gib_per_layer": {"cpu": 1.0}, "io_gib": io_gib})

    def test_only_end_windows_pay_for_their_tables(self):
        m = self.manifest({"cpu": 2.0})
        self.assertEqual(m.window_gib(0, 1, "cpu"), 3.0)
        self.assertEqual(m.window_gib(1, 3, "cpu"), 2.0)
        self.assertEqual(m.window_gib(3, 4, "cpu"), 3.0)
        self.assertEqual(m.window_gib(0, 4, "cpu"), 8.0)
        self.assertEqual(m.windows_fitting(2.5, "cpu"), [(1, 3)])
        self.assertEqual(Manifest.from_dict(m.to_dict()), m)
        self.assertEqual(m.window_gib(0, 1, "unknown"), 3.0)

    def test_bad_costs_are_rejected(self):
        for cost in (-1.0, float("nan"), float("inf"), "1.2", True):
            with self.subTest(cost=cost), self.assertRaises(ManifestError):
                self.manifest({"cpu": cost})


class MetadataOnlyFetch(unittest.TestCase):
    def test_no_weight_payload_download_and_legacy_fallback_still_loads(self):
        import shutil
        from commonnet import hub
        from commonnet.tests.test_hub import make_server, model_files
        from commonnet.weights import CheckpointReader
        for sharded in (True, False):
            with self.subTest(sharded=sharded), tempfile.TemporaryDirectory() as d:
                _, files = model_files(sharded)
                server, log, _ = make_server(files)
                try:
                    h = hub.Hub("org/tiny", endpoint=f"http://127.0.0.1:{server.server_port}",
                                environ={"COMMONNET_CACHE": d}, log=lambda _: None, sleep=lambda _: None)
                    original_dir = h.dir
                    metadata = h.fetch_metadata()
                    self.assertEqual(h.dir, original_dir)
                    reader = CheckpointReader(metadata)
                    self.assertEqual(reader.config.hidden_size, 64)
                    self.assertEqual(caps_for(metadata, 0, 4), ())
                    payloads = [(n, rng) for _, n, rng, _ in log if n.endswith(".safetensors")]
                    if sharded:
                        self.assertEqual(payloads, [])
                    else:
                        self.assertEqual(len(payloads), 2)
                        self.assertTrue(all(rng for _, rng in payloads))
                    self.assertFalse(list(Path(metadata).glob("*.safetensors")))
                    local = h.fetch(client=True)
                    io = EndIO(local, "torch", want_embed=True, want_head=True)
                    self.assertEqual(io.embed([3, 10]).shape, (1, 2, 64))
                    self.assertEqual(len(io.head_ids(torch.ones(1, 2, 64), "last", None, 0)), 1)
                finally:
                    server.shutdown(); server.server_close()
