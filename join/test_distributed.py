"""Offline guards for v0.2 configuration, placement, and CLI compatibility."""
import argparse
import importlib.util
import json
import io
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import runtime
import join

benchmark_spec = importlib.util.spec_from_file_location(
    "distributed_benchmark", Path(__file__).resolve().parents[1] / "bench/distributed.py")
benchmark = importlib.util.module_from_spec(benchmark_spec)
benchmark_spec.loader.exec_module(benchmark)


def listening_endpoint(case):
    """A loopback endpoint on an ephemeral port that accepts TCP connections until the test ends."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    case.addCleanup(server.close)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return f"127.0.0.1:{server.getsockname()[1]}"


def closed_endpoint(case):
    """A loopback endpoint on an ephemeral port that nothing is listening on."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with probe:
        probe.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{probe.getsockname()[1]}"


class RuntimeHealthTests(unittest.TestCase):
    def test_listening_endpoint_is_reachable(self):
        self.assertEqual(runtime.unreachable_rpc_servers([listening_endpoint(self)]), [])

    def test_closed_endpoint_is_reported_and_live_one_is_not(self):
        live, dead = listening_endpoint(self), closed_endpoint(self)
        self.assertEqual(runtime.unreachable_rpc_servers([live, dead]), [dead])

    def test_timeout_counts_as_unreachable_and_uses_short_timeout(self):
        with patch.object(runtime.socket, "create_connection", side_effect=TimeoutError("timed out")) as connect:
            self.assertEqual(runtime.unreachable_rpc_servers(["192.0.2.1:50051"]), ["192.0.2.1:50051"])
        self.assertEqual(connect.call_args.kwargs["timeout"], runtime.RPC_REACHABILITY_TIMEOUT_SECONDS)

    def test_missing_worker_is_detected_from_either_marker(self):
        for line in ("error: Remote RPC server crashed", "rpc: send failed (errno 32)"):
            with self.subTest(line=line):
                self.assertTrue(runtime.missing_worker_in_log(["starting\n", line, "last line"]))

    def test_missing_worker_detection_accepts_one_string(self):
        self.assertTrue(runtime.missing_worker_in_log("ok\nerror: send failed\n"))

    def test_unrelated_log_tail_is_not_a_missing_worker(self):
        self.assertFalse(runtime.missing_worker_in_log(["main: server is listening", "CUDA out of memory"]))
        self.assertFalse(runtime.missing_worker_in_log([]))
        self.assertFalse(runtime.missing_worker_in_log(""))

    def test_missing_worker_detection_does_not_consume_its_input(self):
        tail = ["error: Remote RPC server crashed"]
        runtime.missing_worker_in_log(tail)
        self.assertEqual(tail, ["error: Remote RPC server crashed"])


class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        for filename in ("server", "target.gguf", "draft.gguf"):
            (self.root / filename).touch()
        self.data = {
            "version": 2, "binary": "server", "target_model": "target.gguf",
            "draft_model": "draft.gguf", "model_name": "common-large",
            "rpc_servers": ["127.0.0.1:50051", "127.0.0.1:50052"],
            "devices": ["RPC0", "RPC1"],
        }

    def config(self):
        path = self.root / "network.json"
        path.write_text(json.dumps(self.data))
        return runtime.DistributedConfig(path)

    def test_paths_resolve_against_config_and_alias_is_preserved(self):
        config = self.config()
        args = config.argv()
        self.assertEqual(args[args.index("-m") + 1], str(self.root / "target.gguf"))
        self.assertEqual(args[args.index("--alias") + 1], "common-large")
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertEqual(args[args.index("--port") + 1], "11436")

    def test_target_uses_only_remote_devices_and_draft_is_local(self):
        args = self.config().argv()
        self.assertEqual(args[args.index("--device") + 1], ",".join(self.data["devices"]))
        self.assertEqual(args[args.index("--split-mode") + 1], "layer")
        self.assertEqual(args[args.index("--spec-draft-device") + 1], "none")
        self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "4")
        self.assertEqual(args[args.index("--fit") + 1], "off")

    def test_plain_control_has_same_target_placement(self):
        speculative = self.config().extra_argv()
        del self.data["draft_model"]
        plain = self.config().extra_argv()
        self.assertEqual(plain[:plain.index("--spec-type")], speculative[:speculative.index("--spec-type")])
        self.assertEqual(plain[-2:], ["--spec-type", "none"])
        self.assertNotIn("--spec-draft-model", plain)

    def test_invalid_configs_fail_before_launch(self):
        invalid = [
            ("version", 1), ("unknown", True), ("rpc_servers", ["a:1"]),
            ("rpc_servers", ["a:1", "a:1"]), ("rpc_servers", ["a:0", "b:2"]),
            ("devices", ["MTL0", "MTL1"]),
            ("devices", ["RPC0[other:1]", "RPC0[other:2]"]),
            ("tensor_split", [1, 0]), ("tensor_split", [1]),
            ("tensor_split", [1, float("nan")]), ("draft_tokens", 0),
            ("draft_tokens", True), ("parallel", 0), ("context", "4096"),
            ("draft_device", "RPC0[host:1]"), ("target_model", "missing"),
            ("model_name", ""),
        ]
        original = self.data.copy()
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                self.data = {**original, key: value}
                with self.assertRaises(runtime.FetchFailed):
                    self.config()

    def test_incompatible_binary_fails_preflight(self):
        with patch.object(runtime.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "--rpc", "")):
            with self.assertRaises(runtime.FetchFailed):
                self.config().preflight()

    def test_preflight_failure_is_reported(self):
        with patch.object(runtime.subprocess, "run", side_effect=OSError("cannot execute")):
            with self.assertRaisesRegex(runtime.FetchFailed, "cannot execute"):
                self.config().preflight()

    def test_discovery_rejects_missing_remote_device(self):
        config = self.config()
        help_text = " ".join(config.extra_argv())
        results = [subprocess.CompletedProcess([], 0, help_text, ""),
                   subprocess.CompletedProcess([], 0, "RPC0: 127.0.0.1:50051 (123 MiB)", "")]
        with patch.object(runtime, "unreachable_rpc_servers", return_value=[]), \
                patch.object(runtime.subprocess, "run", side_effect=results):
            with self.assertRaisesRegex(runtime.FetchFailed, "discovery failed"):
                config.preflight()

    def test_discovery_accepts_complete_group(self):
        config = self.config()
        results = [subprocess.CompletedProcess([], 0, " ".join(config.extra_argv()), ""),
                   subprocess.CompletedProcess([], 0, "RPC0: 127.0.0.1:50051 (123 MiB)\nRPC1: 127.0.0.1:50052 (123 MiB)", "")]
        with patch.object(runtime, "unreachable_rpc_servers", return_value=[]), \
                patch.object(runtime.subprocess, "run", side_effect=results):
            config.preflight()

    def test_discovery_rejects_two_devices_on_only_one_peer(self):
        config = self.config()
        results = [subprocess.CompletedProcess([], 0, " ".join(config.extra_argv()), ""),
                   subprocess.CompletedProcess([], 0, "RPC0: 127.0.0.1:50051 (123 MiB)\nRPC1: 127.0.0.1:50051 (123 MiB)", "")]
        with patch.object(runtime, "unreachable_rpc_servers", return_value=[]), \
                patch.object(runtime.subprocess, "run", side_effect=results):
            with self.assertRaisesRegex(runtime.FetchFailed, "discovery failed"):
                config.preflight()

    def test_preflight_names_unreachable_worker_before_discovery(self):
        live, dead = listening_endpoint(self), closed_endpoint(self)
        self.data["rpc_servers"] = [live, dead]
        config = self.config()
        help_text = " ".join(config.extra_argv())
        with patch.object(runtime.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, help_text, "")) as run:
            with self.assertRaisesRegex(runtime.FetchFailed, f"not accepting connections at {dead}") as error:
                config.preflight()
        self.assertNotIn(live, str(error.exception))
        self.assertEqual(run.call_count, 1)

    def test_preflight_accepts_reachable_workers(self):
        first, second = listening_endpoint(self), listening_endpoint(self)
        self.data["rpc_servers"] = [first, second]
        config = self.config()
        results = [subprocess.CompletedProcess([], 0, " ".join(config.extra_argv()), ""),
                   subprocess.CompletedProcess([], 0, f"RPC0: {first} (123 MiB)\nRPC1: {second} (123 MiB)", "")]
        with patch.object(runtime.subprocess, "run", side_effect=results):
            config.preflight()

    def test_service_preserves_config_absolute_path(self):
        args = argparse.Namespace(gateway="http://gateway", model="", name="node",
                                  operator="friend", cost=0, region=None, capability=None,
                                  no_update=True, lan=True, distributed_config=str(self.root / "network.json"))
        command = join._service_argv(args)
        self.assertIn("--distributed-config", command)
        self.assertEqual(command[-1], str(self.root / "network.json"))
        self.assertIn("--no-update", command)
        self.assertNotIn("--model", command)

    def test_checkout_cli_uses_checkout_join_without_download(self):
        spec = importlib.util.spec_from_file_location("common_cli", Path(__file__).resolve().parents[1] / "common/common.py")
        import sys
        cli = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"common_cli": cli}):
            spec.loader.exec_module(cli)
        args = argparse.Namespace(no_update=True, auto=False, model=None, region=None,
                                  rest=["--distributed-config", "network.json"])
        for local_only in (False, True):
            args.rest = (["--local-only"] if local_only else []) + ["--distributed-config", "network.json"]
            with self.subTest(local_only=local_only), patch.object(cli.urllib.request, "urlopen", side_effect=AssertionError("unexpected download")), patch.object(cli.os, "execv") as execute:
                cli.cmd_join_or_serve("join", "http://gateway", args)
            argv = execute.call_args.args[1]
            self.assertEqual(Path(argv[1]), Path(join.__file__).resolve())
            self.assertIn("--no-update", argv)
            self.assertEqual(argv[-len(args.rest):], args.rest)
            self.assertEqual("--gateway" in argv, not local_only)

    def test_benchmark_counts_usage_instead_of_sse_chunks(self):
        events = [
            {"choices": [{"delta": {"content": "hello "}}]},
            {"choices": [{"delta": {"content": "world"}}]},
            {"choices": [], "usage": {"completion_tokens": 5}},
        ]
        stream = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
        with patch.object(benchmark, "request", return_value=io.BytesIO(stream)):
            result = benchmark.generate("http://local", "common-large", "hi", 8)
        self.assertEqual(result["completion_tokens"], 5)
        self.assertEqual(result["output"], "hello world")
        self.assertGreater(result["end_to_end_tokens_per_second"], 0)

    def test_benchmark_missing_usage_is_not_invented(self):
        stream = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
        with patch.object(benchmark, "request", return_value=io.BytesIO(stream)):
            result = benchmark.generate("http://local", "common-large", "hi", 8)
        self.assertIsNone(result["end_to_end_tokens_per_second"])

    def test_benchmark_surfaces_stream_errors(self):
        with patch.object(benchmark, "request", return_value=io.BytesIO(b'data: {"error":"RPC disconnected"}\n')):
            with self.assertRaisesRegex(RuntimeError, "RPC disconnected"):
                benchmark.generate("http://local", "common-large", "hi", 8)

    def test_benchmark_rejects_truncated_stream(self):
        with patch.object(benchmark, "request", return_value=io.BytesIO(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n')):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                benchmark.generate("http://local", "common-large", "hi", 8)

    def test_local_only_main_never_enters_registration_or_updates(self):
        config = self.config()
        with patch.object(join.sys, "argv", ["join.py", "--local-only", "--distributed-config", str(config.path)]), \
                patch.object(join.sys, "stdout"), \
                patch.object(join, "run_local_coordinator") as run, \
                patch.object(join, "read_identity", side_effect=AssertionError("identity read")), \
                patch.object(join, "resolve_model", side_effect=AssertionError("catalogue")), \
                patch.object(join, "refresh_siblings", side_effect=AssertionError("update")), \
                patch.object(join, "start_tunnel", side_effect=AssertionError("tunnel")), \
                patch.object(join.worker, "serve", side_effect=AssertionError("worker")), \
                patch.object(join.urllib.request, "urlopen", side_effect=AssertionError("gateway")):
            join.main()
        run.assert_called_once_with(str(config.path))

    def test_local_only_rejects_registration_related_flags(self):
        for extra in (["--lan"], ["--permanent"], ["--list-catalogue"], ["--model", "other"]):
            with self.subTest(extra=extra), patch.object(join.sys, "argv", ["join.py", "--local-only", "--distributed-config", "unused", *extra]), \
                    patch.object(join.sys, "stdout"), patch.object(join.sys, "stderr"), \
                    patch.object(join, "run_local_coordinator") as run:
                with self.assertRaises(SystemExit) as error:
                    join.main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_local_only_interrupt_stops_only_its_model_child(self):
        config = self.config()
        child = MagicMock()
        child.wait.side_effect = KeyboardInterrupt
        with patch.object(join, "start_llama_server", return_value=child), \
                patch.object(join, "wait_for_llama", return_value=True), \
                patch.object(join, "stop_process") as stop, \
                patch.object(join.signal, "signal"), patch.object(join.sys, "stdout"):
            join.run_local_coordinator(str(config.path))
        stop.assert_called_once_with(child)

    def test_local_only_startup_failure_stops_child(self):
        config = self.config()
        child = MagicMock()
        with patch.object(join, "start_llama_server", return_value=child), \
                patch.object(join, "wait_for_llama", return_value=False), \
                patch.object(join, "LOCAL_RESTART_DELAYS_SECONDS", (0,)), \
                patch.object(join, "stop_process") as stop, \
                patch.object(join.signal, "signal"), patch.object(join.sys, "stderr"):
            with self.assertRaises(SystemExit):
                join.run_local_coordinator(str(config.path))
        # Each failed attempt and the final cleanup stop the same child, so
        # the check is that nothing but that child is ever stopped.
        stop.assert_called_with(child)
        self.assertTrue(all(call.args == (child,) for call in stop.call_args_list))


if __name__ == "__main__":
    unittest.main()
