"""Offline guards for v0.2 configuration, placement, and CLI compatibility."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

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

    # Captured from the local llama.cpp build on Apple M3 (`--rpc 127.0.0.1:50051 --list-devices`).
    # The local accelerator is listed with no RPC prefix; the worker line has an address
    # and no backend name, so the parser must not infer one.
    REAL_LOCAL_LISTING = ("Available devices:\n"
                          "  MTL0: Apple M3 (12124 MiB, 12123 MiB free)\n")
    REAL_RPC_LISTING = REAL_LOCAL_LISTING + "  RPC0: 127.0.0.1:50051 (12124 MiB, 12123 MiB free)\n"

    def real_preflight(self, listing, **overrides):
        self.data.update(overrides)
        config = self.config()
        results = [subprocess.CompletedProcess([], 0, " ".join(config.extra_argv()), ""),
                   subprocess.CompletedProcess([], 0, listing, "")]
        with patch.object(runtime, "unreachable_rpc_servers", return_value=[]), \
                patch.object(runtime.subprocess, "run", side_effect=results):
            config.preflight()

    def test_discovery_reads_real_worker_listing(self):
        listing = self.REAL_RPC_LISTING + "  RPC1: 127.0.0.1:50052 (12124 MiB, 12123 MiB free)\n"
        self.real_preflight(listing)

    def test_real_local_accelerator_is_a_valid_draft_device(self):
        listing = self.REAL_RPC_LISTING + "  RPC1: 127.0.0.1:50052 (12124 MiB, 12123 MiB free)\n"
        self.real_preflight(listing, draft_device="MTL0")

    def test_remote_worker_is_not_a_local_draft_device(self):
        # RPC0 is listed by the real binary, but it is a remote worker, not a local device.
        listing = self.REAL_RPC_LISTING + "  RPC1: 127.0.0.1:50052 (12124 MiB, 12123 MiB free)\n"
        with self.assertRaisesRegex(runtime.FetchFailed, "local draft device RPC0 is unavailable"):
            self.real_preflight(listing, draft_device="RPC0")

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
        run.assert_called_once_with(str(config.path), 11436)

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

    def test_local_only_port_reaches_runtime_and_health_check(self):
        child = MagicMock()
        child.poll.return_value = None
        child.wait.return_value = 0
        with patch.object(join, "start_llama_server", return_value=child) as start, \
                patch.object(join, "wait_for_llama", return_value=True) as wait, \
                patch.object(join, "LOCAL_RESTART_DELAYS_SECONDS", (0,)), \
                patch.object(join, "die", side_effect=SystemExit), \
                patch.object(join, "stop_process"):
            with self.assertRaises(SystemExit):
                join.run_local_coordinator(str(self.config().path), 11437)
        self.assertEqual(start.call_args_list[0].kwargs["port"], 11437)
        self.assertEqual(wait.call_args_list[0].kwargs["port"], 11437)

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


class LlamaDownloadCheckTests(unittest.TestCase):
    """The llama.cpp archive is checked against GitHub's sha256 for it, or it is not installed."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name, value in (("DOWNLOAD_CACHE", self.dir / "cache"), ("RUNTIME_DIR", self.dir / "runtime"),
                            ("CURRENT_RUNTIME", self.dir / "runtime" / "current")):
            patcher = patch.object(runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def release_json(self, assets):
        return json.dumps({"tag_name": "b1", "assets": assets})

    def test_digest_comes_from_the_release_record_for_the_named_asset(self):
        digest = "ab" * 32
        body = self.release_json([{"name": "other.zip", "digest": "sha256:" + "11" * 32},
                                  {"name": "llama-b1-bin-x.tar.gz", "digest": "sha256:" + digest}])
        with patch.object(runtime, "_read_text", return_value=body):
            self.assertEqual(runtime.llama_asset_digest("b1", "llama-b1-bin-x.tar.gz"), digest)

    def test_missing_or_malformed_digest_is_none(self):
        cases = [self.release_json([{"name": "llama-b1-bin-x.tar.gz"}]),
                 self.release_json([{"name": "llama-b1-bin-x.tar.gz", "digest": "md5:abc"}]),
                 self.release_json([{"name": "llama-b1-bin-x.tar.gz", "digest": "sha256:xyz"}]),
                 self.release_json([]), "[]", "not json"]
        for body in cases:
            with self.subTest(body=body):
                with patch.object(runtime, "_read_text", return_value=body):
                    self.assertIsNone(runtime.llama_asset_digest("b1", "llama-b1-bin-x.tar.gz"))

    def test_unreachable_api_is_none_not_an_exception(self):
        with patch.object(runtime, "_read_text", side_effect=runtime.urllib.error.URLError("offline")):
            self.assertIsNone(runtime.llama_asset_digest("b1", "llama-b1-bin-x.tar.gz"))

    def test_install_refuses_when_no_checksum_is_recorded(self):
        with patch.object(runtime, "_installed_binary", return_value=None), \
                patch.object(runtime, "llama_asset_name", return_value="llama-b1-bin-x.tar.gz"), \
                patch.object(runtime, "resolve_llama_build", return_value="b1"), \
                patch.object(runtime, "llama_asset_digest", return_value=None), \
                patch.object(runtime, "_download") as download:
            with self.assertRaisesRegex(runtime.FetchFailed, "no checksum"):
                runtime.ensure_llama_server()
        download.assert_not_called()

    def test_download_is_made_with_the_recorded_digest(self):
        digest = "cd" * 32
        with patch.object(runtime, "_installed_binary", return_value=None), \
                patch.object(runtime, "llama_asset_name", return_value="llama-b1-bin-x.tar.gz"), \
                patch.object(runtime, "resolve_llama_build", return_value="b1"), \
                patch.object(runtime, "llama_asset_digest", return_value=digest), \
                patch.object(runtime, "_download", side_effect=runtime.FetchFailed("stop here")) as download:
            with self.assertRaises(runtime.FetchFailed):
                runtime.ensure_llama_server()
        self.assertEqual(download.call_args.kwargs["expected_sha"], digest)

    def test_cached_archive_with_the_wrong_bytes_is_discarded_before_redownload(self):
        cache = self.dir / "cache"
        cache.mkdir(parents=True)
        (cache / "llama-b1-bin-x.tar.gz").write_bytes(b"tampered")
        seen = {}

        def fake_download(url, dest, **kwargs):
            seen["existed"] = dest.exists()
            raise runtime.FetchFailed("stop here")

        with patch.object(runtime, "_installed_binary", return_value=None), \
                patch.object(runtime, "llama_asset_name", return_value="llama-b1-bin-x.tar.gz"), \
                patch.object(runtime, "resolve_llama_build", return_value="b1"), \
                patch.object(runtime, "llama_asset_digest", return_value=runtime.hashlib.sha256(b"good").hexdigest()), \
                patch.object(runtime, "_download", side_effect=fake_download):
            with self.assertRaises(runtime.FetchFailed):
                runtime.ensure_llama_server()
        self.assertFalse(seen["existed"])

    def test_download_with_a_wrong_checksum_keeps_nothing(self):
        source = self.dir / "served.bin"
        source.write_bytes(b"llama bytes")
        dest = self.dir / "archive.tar.gz"
        with self.assertRaisesRegex(runtime.FetchFailed, "did not match its recorded checksum"):
            runtime._download_once(source.as_uri(), dest, expected_sha="00" * 32)
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_download_with_the_right_checksum_is_kept(self):
        source = self.dir / "served.bin"
        source.write_bytes(b"llama bytes")
        dest = self.dir / "archive.tar.gz"
        runtime._download_once(source.as_uri(), dest, expected_sha=runtime.hashlib.sha256(b"llama bytes").hexdigest())
        self.assertEqual(dest.read_bytes(), b"llama bytes")


class RpcServerTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_rpc_server_is_found_beside_the_installed_llama_server(self):
        (self.dir / "llama-server").write_text("")
        rpc = self.dir / runtime.RPC_SERVER_NAME
        rpc.write_text("")
        with patch.object(runtime, "ensure_llama_server", return_value=self.dir / "llama-server"):
            self.assertEqual(runtime.ensure_rpc_server(), rpc)

    def test_build_without_rpc_server_is_refused_with_the_remedy(self):
        (self.dir / "llama-server").write_text("")
        with patch.object(runtime, "ensure_llama_server", return_value=self.dir / "llama-server"):
            with self.assertRaisesRegex(runtime.FetchFailed, "without RPC support"):
                runtime.ensure_rpc_server()


def zip_of(names):
    """A small zip archive holding `names`, standing in for a release archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"bytes of " + name.encode())
    return buf.getvalue()


class NvidiaDetectionTests(unittest.TestCase):
    """nvidia-smi decides whether a Windows machine gets the CUDA build, and which line."""

    def detect(self, listing, report=None, missing=False):
        def fake_run(args, **kwargs):
            if missing:
                raise FileNotFoundError("nvidia-smi")
            rc, out = listing if args == ["nvidia-smi", "-L"] else report
            return subprocess.CompletedProcess(args, rc, stdout=out, stderr="")

        with patch.object(runtime.subprocess, "run", side_effect=fake_run):
            return runtime.detect_nvidia_gpu()

    def test_no_nvidia_smi_means_no_gpu(self):
        self.assertIsNone(self.detect(None, missing=True))

    def test_nonzero_listing_or_no_device_line_means_no_gpu(self):
        self.assertIsNone(self.detect((9, ""), (0, "CUDA Version: 13.2")))
        self.assertIsNone(self.detect((0, "No devices were found\n"), (0, "CUDA Version: 13.2")))

    def test_device_line_with_driver_cuda_13_is_read(self):
        gpu = self.detect((0, "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-x)\n"),
                          (0, "| NVIDIA-SMI 575.00  Driver Version: 575.00  CUDA Version: 13.2 |"))
        self.assertEqual(gpu, runtime.NvidiaGpu(driver_cuda=(13, 2)))

    def test_device_line_with_driver_cuda_12_is_read(self):
        gpu = self.detect((0, "GPU 0: NVIDIA A10 (UUID: GPU-x)\n"),
                          (0, "| Driver Version: 550.90  CUDA Version: 12.4 |"))
        self.assertEqual(gpu, runtime.NvidiaGpu(driver_cuda=(12, 4)))

    def test_unreadable_version_is_a_gpu_with_no_version(self):
        self.assertEqual(self.detect((0, "GPU 0: x\n"), (0, "no version in this report")),
                         runtime.NvidiaGpu(driver_cuda=None))
        self.assertEqual(self.detect((0, "GPU 0: x\n"), (9, "")), runtime.NvidiaGpu(driver_cuda=None))


class WindowsCudaSelectionTests(unittest.TestCase):
    def test_newest_line_the_driver_supports_is_chosen(self):
        cases = [((13, 9), "13.4"), ((13, 4), "13.4"), ((13, 3), "12.4"), ((13, 0), "12.4"),
                 ((12, 8), "12.4"), ((12, 4), "12.4"), (None, "12.4")]
        for driver, line in cases:
            with self.subTest(driver=driver):
                self.assertEqual(runtime.windows_cuda_line(runtime.NvidiaGpu(driver_cuda=driver)), line)

    def test_driver_older_than_every_line_is_refused_not_given_the_cpu_build(self):
        with self.assertRaisesRegex(runtime.FetchFailed, "older than the 12.4 build"):
            runtime.windows_cuda_line(runtime.NvidiaGpu(driver_cuda=(11, 8)))

    def test_cuda_archive_pair_names(self):
        self.assertEqual(runtime.windows_cuda_assets("b1", "13.4"),
                         ("llama-b1-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"))

    def test_windows_cpu_asset_is_unchanged(self):
        self.assertEqual(runtime.llama_asset_name("b1", "Windows", "AMD64"), "llama-b1-bin-win-cpu-x64.zip")


class WindowsCudaInstallTests(unittest.TestCase):
    """Windows with an NVIDIA GPU installs the CUDA pair, verified, into one directory.

    Nothing here touches the network: the GPU probe, the platform, the digests and
    the downloads are all stubbed, and `_runnable` is stubbed because the test
    archives hold no real binaries.
    """
    BUILD = "b1"

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name, value in (("DOWNLOAD_CACHE", self.dir / "cache"), ("RUNTIME_DIR", self.dir / "runtime"),
                            ("CURRENT_RUNTIME", self.dir / "runtime" / "current"),
                            ("RPC_SERVER_NAME", "ggml-rpc-server.exe")):
            patcher = patch.object(runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        cuda = zip_of(["llama-server.exe", "ggml-rpc-server.exe", "ggml-cuda.dll", "ggml-rpc.dll"])
        cudart = zip_of(["cudart64_13.dll", "cublas64_13.dll"])
        cpu = zip_of(["llama-server.exe", "ggml-rpc-server.exe"])
        self.served = {runtime.llama_asset_name(self.BUILD, "Windows", "AMD64"): cpu}
        for line in ("13.4", "12.4"):
            cuda_name, cudart_name = runtime.windows_cuda_assets(self.BUILD, line)
            self.served[cuda_name] = cuda
            self.served[cudart_name] = cudart
        self.digests = {name: hashlib.sha256(data).hexdigest() for name, data in self.served.items()}

    @contextmanager
    def windows(self, gpu):
        """Windows, with `gpu` as the probe result. Yields the list of downloads made."""
        downloads = []

        def fake_download(url, dest, expected_sha=None, what=None):
            name = url.rsplit("/", 1)[1]
            downloads.append((name, expected_sha))
            if name not in self.served:
                raise runtime.FetchFailed(f"{name} is not served")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.served[name])

        with patch.object(runtime.platform, "system", return_value="Windows"), \
                patch.object(runtime.platform, "machine", return_value="AMD64"), \
                patch.object(runtime, "detect_nvidia_gpu", return_value=gpu), \
                patch.object(runtime, "resolve_llama_build", return_value=self.BUILD), \
                patch.object(runtime, "llama_asset_digest", side_effect=lambda build, asset: self.digests.get(asset)), \
                patch.object(runtime, "_download", side_effect=fake_download), \
                patch.object(runtime, "_runnable", return_value=True):
            yield downloads

    def test_nvidia_13_4_driver_installs_the_13_4_pair_into_one_directory(self):
        with self.windows(runtime.NvidiaGpu(driver_cuda=(13, 4))) as downloads:
            binary = runtime.ensure_llama_server()
            rpc = runtime.ensure_rpc_server()
        names = [name for name, _ in downloads]
        self.assertEqual(names, list(runtime.windows_cuda_assets(self.BUILD, "13.4")))
        self.assertEqual([sha for _, sha in downloads], [self.digests[n] for n in names])
        self.assertEqual(binary.parent, self.dir / "runtime" / "b1-cuda-13.4")
        self.assertEqual(rpc, binary.parent / "ggml-rpc-server.exe")
        self.assertTrue((binary.parent / "ggml-cuda.dll").is_file())
        self.assertTrue((binary.parent / "cudart64_13.dll").is_file())
        self.assertEqual(runtime.CURRENT_RUNTIME.read_text(), "b1-cuda-13.4")

    def test_driver_below_13_4_gets_the_12_4_pair(self):
        for driver in ((12, 8), (13, 0), (13, 3)):
            with self.subTest(driver=driver):
                with self.windows(runtime.NvidiaGpu(driver_cuda=driver)) as downloads:
                    binary = runtime.ensure_llama_server()
                self.assertEqual([n for n, _ in downloads], list(runtime.windows_cuda_assets(self.BUILD, "12.4")))
                self.assertEqual(binary.parent.name, "b1-cuda-12.4")
                self.assertEqual(runtime.CURRENT_RUNTIME.read_text(), "b1-cuda-12.4")
                # Fresh state for the next driver: no cached archive, no install.
                shutil.rmtree(self.dir / "cache", ignore_errors=True)
                shutil.rmtree(self.dir / "runtime", ignore_errors=True)

    def test_unreadable_driver_version_gets_the_12_4_pair(self):
        with self.windows(runtime.NvidiaGpu(driver_cuda=None)) as downloads:
            binary = runtime.ensure_llama_server()
        self.assertEqual([n for n, _ in downloads], list(runtime.windows_cuda_assets(self.BUILD, "12.4")))
        self.assertEqual(binary.parent.name, "b1-cuda-12.4")

    def test_no_nvidia_gpu_keeps_the_cpu_build(self):
        with self.windows(None) as downloads:
            binary = runtime.ensure_llama_server()
            rpc = runtime.ensure_rpc_server()
        self.assertEqual([n for n, _ in downloads], ["llama-b1-bin-win-cpu-x64.zip"])
        self.assertEqual(binary.parent.name, "b1")
        self.assertEqual(rpc, binary.parent / "ggml-rpc-server.exe")
        self.assertEqual(runtime.CURRENT_RUNTIME.read_text(), "b1")

    def test_cpu_install_is_replaced_when_a_gpu_appears(self):
        cpu_dir = self.dir / "runtime" / "b1"
        cpu_dir.mkdir(parents=True)
        (cpu_dir / "llama-server.exe").write_text("")
        runtime.CURRENT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        runtime.CURRENT_RUNTIME.write_text("b1")
        with self.windows(runtime.NvidiaGpu(driver_cuda=(13, 4))) as downloads:
            binary = runtime.ensure_llama_server()
        self.assertEqual(binary.parent.name, "b1-cuda-13.4")
        self.assertNotIn("win-cpu", " ".join(n for n, _ in downloads))

    def test_installed_cpu_build_is_kept_when_there_is_no_gpu(self):
        cpu_dir = self.dir / "runtime" / "b1"
        cpu_dir.mkdir(parents=True)
        (cpu_dir / "llama-server.exe").write_text("")
        runtime.CURRENT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        runtime.CURRENT_RUNTIME.write_text("b1")
        with self.windows(None) as downloads:
            binary = runtime.ensure_llama_server()
        self.assertEqual(binary, cpu_dir / "llama-server.exe")
        self.assertEqual(downloads, [])

    def test_cuda_download_failure_never_falls_back_to_the_cpu_build(self):
        del self.served[runtime.windows_cuda_assets(self.BUILD, "13.4")[0]]
        with self.windows(runtime.NvidiaGpu(driver_cuda=(13, 4))) as downloads:
            with self.assertRaisesRegex(runtime.FetchFailed, "cuda"):
                runtime.ensure_llama_server()
        self.assertNotIn("win-cpu", " ".join(n for n, _ in downloads))

    def test_cuda_install_without_its_gpu_backend_is_refused(self):
        install = self.dir / "runtime" / "b1-cuda-13.4"
        install.mkdir(parents=True)
        (install / "llama-server.exe").write_text("")
        (install / "ggml-rpc-server.exe").write_text("")
        runtime.CURRENT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        runtime.CURRENT_RUNTIME.write_text("b1-cuda-13.4")
        with self.windows(runtime.NvidiaGpu(driver_cuda=(13, 4))) as downloads:
            with self.assertRaisesRegex(runtime.FetchFailed, "would run on the CPU"):
                runtime.ensure_rpc_server()
        self.assertEqual(downloads, [])


if __name__ == "__main__":
    unittest.main()
