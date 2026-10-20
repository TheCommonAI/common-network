"""Offline tests for generating a distributed group's network.json.

Nothing here opens a socket or starts llama.cpp. The model and draft files are
empty stand-ins, and every test runs with socket.create_connection forbidden.
"""
import contextlib
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import group
import group_config
import runtime


def member(rpc_port: int) -> dict:
    return {"verifier": "ab" * 32, "rpc_port": rpc_port, "enrolled": "2026-10-20T00:00:00Z"}


def make_group(groups_dir: Path, name: str = "lab", members: dict | None = None) -> None:
    data = {"v": 1, "group": name, "relay": "127.0.0.1:7443", "cert_fp": "cd" * 32,
            "secret": "ef" * 32, "members": members or {}, "used": {}}
    group.GroupState(groups_dir / name / "group.json", data).save()


class NoNetworkTestCase(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket, "create_connection",
                             side_effect=AssertionError("config generation must not connect"))
        guard.start()
        self.addCleanup(guard.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.groups_dir = self.root / "groups"
        for filename in ("llama-server", "Qwen_Qwen3-32B-Q3_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf"):
            (self.root / filename).touch()
        self.template = json.loads(group_config.TEMPLATE.read_text())


class BuildConfigTests(NoNetworkTestCase):
    def build(self, endpoints, template=None, **overrides):
        options = {"binary": str(self.root / "llama-server"),
                   "target": str(self.root / "Qwen_Qwen3-32B-Q3_K_M.gguf"),
                   "drafter": str(self.root / "Qwen3-0.6B-Q8_0.gguf"),
                   "backends": ["MTL0"] * min(1, len(endpoints)) + ["CUDA0"] * max(0, len(endpoints) - 1)}
        options.update(overrides)
        return group_config.build_config(template or self.template, endpoints=endpoints, **options)

    def test_cpu_device_is_refused_in_either_position(self):
        endpoints = ["127.0.0.1:50051", "127.0.0.1:52100"]
        for backends in (["CPU", "CUDA0"], ["MTL0", "CPU0"]):
            with self.subTest(backends=backends):
                with self.assertRaisesRegex(group_config.ConfigError, "GPU-only"):
                    self.build(endpoints, backends=backends)

    def test_endpoint_without_a_gpu_backend_is_refused(self):
        endpoints = ["127.0.0.1:50051", "127.0.0.1:52100"]
        for backends in (["MTL0", "cuda0"], ["MTL0", "Foo0"], ["MTL0", ""], ["MTL", "CUDA0"]):
            with self.subTest(backends=backends):
                with self.assertRaisesRegex(group_config.ConfigError, "has no GPU device"):
                    self.build(endpoints, backends=backends)

    def test_each_gpu_backend_is_accepted_for_the_member(self):
        endpoints = ["127.0.0.1:50051", "127.0.0.1:52100"]
        for member_backend in ("CUDA0", "Vulkan0", "MTL1"):
            with self.subTest(member_backend=member_backend):
                config = self.build(endpoints, backends=["MTL0", member_backend])
                self.assertEqual(config["devices"], ["RPC0", "RPC1"])

    def test_backend_count_must_match_the_endpoints(self):
        endpoints = ["127.0.0.1:50051", "127.0.0.1:52100"]
        with self.assertRaisesRegex(group_config.ConfigError, "declared GPU backend"):
            self.build(endpoints, backends=["MTL0"])

    def test_drafter_on_cpu_is_refused(self):
        template = {**self.template, "draft_device": "none"}
        with self.assertRaisesRegex(group_config.ConfigError, "not a GPU"):
            self.build(["127.0.0.1:50051", "127.0.0.1:52100"], template=template)

    def test_two_endpoints_fill_servers_devices_and_the_weighted_default_split(self):
        config = self.build(["127.0.0.1:50051", "127.0.0.1:52100"])
        self.assertEqual(config["rpc_servers"], ["127.0.0.1:50051", "127.0.0.1:52100"])
        self.assertEqual(config["devices"], ["RPC0", "RPC1"])
        self.assertEqual(config["tensor_split"], [0.65, 0.35])

    def test_device_order_follows_endpoint_order(self):
        config = self.build(["127.0.0.1:52100", "127.0.0.1:50051"])
        self.assertEqual(config["rpc_servers"], ["127.0.0.1:52100", "127.0.0.1:50051"])
        self.assertEqual(config["devices"], ["RPC0", "RPC1"])

    def test_more_than_two_endpoints_is_refused_for_this_trial(self):
        with self.assertRaisesRegex(group_config.ConfigError, "exactly 2 endpoints"):
            self.build(["127.0.0.1:50051", "127.0.0.1:52100", "127.0.0.1:52101"])

    def test_split_takes_one_weight_per_device(self):
        endpoints = ["127.0.0.1:50051", "127.0.0.1:52100"]
        for split in ((0.65,), (0.5, 0.3, 0.2), ()):
            with self.subTest(split=split):
                with self.assertRaisesRegex(group_config.ConfigError, "one weight per device"):
                    self.build(endpoints, split=split)

    def test_split_weights_must_be_positive_and_finite(self):
        endpoints = ["127.0.0.1:50051", "127.0.0.1:52100"]
        for split in ((0.0, 1.0), (-0.5, 1.5), (float("nan"), 1.0), (float("inf"), 1.0), ("0.5", 0.5)):
            with self.subTest(split=split):
                with self.assertRaisesRegex(group_config.ConfigError, "positive finite"):
                    self.build(endpoints, split=split)

    def test_custom_split_is_written_as_given(self):
        config = self.build(["127.0.0.1:50051", "127.0.0.1:52100"], split=(0.6, 0.4))
        self.assertEqual(config["tensor_split"], [0.6, 0.4])

    def test_fewer_than_two_endpoints_is_refused_and_says_why(self):
        for endpoints in ([], ["127.0.0.1:50051"]):
            with self.subTest(endpoints=endpoints):
                with self.assertRaises(group_config.ConfigError) as caught:
                    self.build(endpoints)
                message = str(caught.exception)
                self.assertIn("at least 2", message)
                self.assertIn("one device is not a distributed run", message)

    def test_duplicate_endpoint_is_refused(self):
        with self.assertRaisesRegex(group_config.ConfigError, "duplicate RPC endpoint"):
            self.build(["127.0.0.1:50051", "127.0.0.1:50051"])

    def test_template_values_carry_through_with_the_trial_choices(self):
        config = self.build(["127.0.0.1:50051", "127.0.0.1:52100"])
        self.assertEqual(config["model_name"], "Qwen3-32B")
        self.assertEqual(config["draft_device"], "MTL0")
        self.assertEqual(config["draft_tokens"], 4)
        self.assertEqual(config["context"], 4096)
        self.assertEqual(config["parallel"], 1)
        self.assertEqual(config["version"], 2)

    def test_draft_length_override_replaces_the_template_value(self):
        config = self.build(["127.0.0.1:50051", "127.0.0.1:52100"], draft_tokens=8)
        self.assertEqual(config["draft_tokens"], 8)


class MemberEndpointTests(NoNetworkTestCase):
    def test_members_come_from_group_json_in_port_order(self):
        make_group(self.groups_dir, members={"patrick": member(52101), "node-a": member(52100)})
        self.assertEqual(group_config.member_endpoints(self.groups_dir, "lab"),
                         ["127.0.0.1:52100", "127.0.0.1:52101"])

    def test_group_without_members_gives_no_endpoints(self):
        make_group(self.groups_dir)
        self.assertEqual(group_config.member_endpoints(self.groups_dir, "lab"), [])

    def test_missing_group_is_a_config_error(self):
        with self.assertRaisesRegex(group_config.ConfigError, "no group at"):
            group_config.member_endpoints(self.groups_dir, "lab")

    def test_invalid_group_name_is_a_config_error(self):
        with self.assertRaises(group_config.ConfigError):
            group_config.member_endpoints(self.groups_dir, "Not A Group")


class CommandTests(NoNetworkTestCase):
    def run_main(self, *extra, group_name="lab", target=None, member_backend="CUDA0"):
        argv = ["--group", group_name, "--groups-dir", str(self.groups_dir),
                "--target", target or str(self.root / "Qwen_Qwen3-32B-Q3_K_M.gguf"),
                "--drafter", str(self.root / "Qwen3-0.6B-Q8_0.gguf"),
                "--binary", str(self.root / "llama-server")]
        if member_backend:
            argv += ["--member-backend", member_backend]
        argv += list(extra)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = group_config.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_writes_a_config_the_runtime_accepts(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        network = self.groups_dir / "lab" / "network.json"
        self.assertTrue(network.is_file())
        self.assertFalse(network.with_name("network.json.tmp").exists())
        config = runtime.DistributedConfig(network)
        self.assertEqual(config.peers, ["127.0.0.1:50051", "127.0.0.1:52100"])
        self.assertEqual(config.devices, ["RPC0", "RPC1"])
        self.assertEqual(config.split, [0.65, 0.35])
        self.assertEqual(config.draft_device, "MTL0")
        self.assertEqual(config.draft_tokens, 4)
        self.assertEqual(config.name, "Qwen3-32B")
        self.assertEqual(config.target, (self.root / "Qwen_Qwen3-32B-Q3_K_M.gguf").resolve())
        self.assertIn(str(network), out)

    def test_second_member_is_refused_and_the_two_device_config_is_kept(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        self.assertEqual(self.run_main()[0], 0)
        network = self.groups_dir / "lab" / "network.json"
        before = network.read_text()
        make_group(self.groups_dir, members={"patrick": member(52100), "node-b": member(52101)})
        code, _, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("exactly 2 endpoints", err)
        self.assertEqual(network.read_text(), before)

    def test_tensor_split_flag_sets_the_weights(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        self.assertEqual(self.run_main("--tensor-split", "0.6,0.4")[0], 0)
        self.assertEqual(runtime.DistributedConfig(self.groups_dir / "lab" / "network.json").split, [0.6, 0.4])

    def test_cpu_member_backend_is_refused_from_the_command_line(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        code, _, err = self.run_main(member_backend="CPU0")
        self.assertEqual(code, 1)
        self.assertIn("GPU-only", err)
        self.assertFalse((self.groups_dir / "lab" / "network.json").exists())

    def test_cpu_local_backend_is_refused_from_the_command_line(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        code, _, err = self.run_main("--local-backend", "CPU")
        self.assertEqual(code, 1)
        self.assertIn("GPU-only", err)
        self.assertFalse((self.groups_dir / "lab" / "network.json").exists())

    def test_members_without_a_declared_backend_are_refused(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        code, _, err = self.run_main(member_backend=None)
        self.assertEqual(code, 1)
        self.assertIn("--member-backend", err)
        self.assertFalse((self.groups_dir / "lab" / "network.json").exists())

    def test_malformed_tensor_split_flag_is_refused(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        code, _, err = self.run_main("--tensor-split", "0.6;0.4")
        self.assertEqual(code, 1)
        self.assertIn("comma-separated numbers", err)
        self.assertFalse((self.groups_dir / "lab" / "network.json").exists())

    def test_no_members_refuses_and_writes_nothing(self):
        make_group(self.groups_dir)
        code, _, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("at least 2", err)
        self.assertFalse((self.groups_dir / "lab" / "network.json").exists())

    def test_missing_target_keeps_the_previous_config(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        self.assertEqual(self.run_main()[0], 0)
        network = self.groups_dir / "lab" / "network.json"
        before = network.read_text()
        code, _, err = self.run_main(target=str(self.root / "missing.gguf"))
        self.assertEqual(code, 1)
        self.assertIn("target_model does not exist", err)
        self.assertEqual(network.read_text(), before)
        self.assertFalse(network.with_name("network.json.tmp").exists())

    def test_unknown_group_is_reported_not_raised(self):
        code, _, err = self.run_main(group_name="nope")
        self.assertEqual(code, 1)
        self.assertIn("no group at", err)

    def test_draft_tokens_flag_reaches_the_written_config(self):
        make_group(self.groups_dir, members={"patrick": member(52100)})
        self.assertEqual(self.run_main("--draft-tokens", "6")[0], 0)
        self.assertEqual(runtime.DistributedConfig(self.groups_dir / "lab" / "network.json").draft_tokens, 6)


class TemplateTests(NoNetworkTestCase):
    def test_template_is_the_qwen3_32b_trial_and_has_only_allowed_fields(self):
        self.assertEqual(self.template["model_name"], "Qwen3-32B")
        self.assertEqual(self.template["draft_device"], "MTL0")
        self.assertEqual(self.template["draft_tokens"], 4)
        self.assertNotIn("rpc_servers", self.template)
        self.assertNotIn("devices", self.template)
        self.assertNotIn("tensor_split", self.template)
        self.assertLessEqual(set(self.template), {"version", "binary", "target_model", "draft_model",
                                                  "model_name", "draft_device", "draft_tokens",
                                                  "context", "parallel"})


if __name__ == "__main__":
    unittest.main()
