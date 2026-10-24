#!/usr/bin/env python3
"""Tests for layers.py and join.py's --network option. Offline: no pip, no download, no node is started."""
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import join
import layers

JOIN_PY = Path(__file__).resolve().parent / "join.py"


class PlatformChoice(unittest.TestCase):
    def test_apple_silicon_is_mac_arm64(self):
        self.assertEqual(layers.platform_kind("Darwin", "arm64"), "mac-arm64")

    def test_nvidia_on_windows_and_linux(self):
        for system in ("Windows", "Linux"):
            self.assertEqual(layers.platform_kind(system, "AMD64", nvidia=True), "nvidia")

    def test_everything_else_is_refused_with_a_reason(self):
        for system, machine, nvidia in (("Linux", "x86_64", False), ("Windows", "AMD64", False), ("Darwin", "x86_64", False)):
            with self.subTest(system=system, machine=machine):
                with self.assertRaises(layers.LayersError) as caught:
                    layers.platform_kind(system, machine, nvidia=nvidia)
                self.assertRegex(str(caught.exception), "GPU|supported")

    def test_requirements_are_pinned_where_it_matters(self):
        plain, extra = layers.requirements("mac-arm64")
        self.assertEqual(extra, [])
        self.assertIn("transformers==5.8.0", plain)  # commonnet/block.py refuses other versions
        self.assertTrue(any(r.startswith("mlx") for r in plain))
        plain, extra = layers.requirements("nvidia")
        self.assertTrue(any(r.startswith("bitsandbytes") for r in plain))
        self.assertIn("--index-url", extra)
        with self.assertRaises(layers.LayersError):
            layers.requirements("riscv")


class Commands(unittest.TestCase):
    def test_build_command(self):
        self.assertEqual(layers.build_command("/py", "https://seed", ["--port", "5"]),
                         ["/py", "-m", "commonnet.node", "--network", "https://seed", "--port", "5"])
        self.assertEqual(layers.build_command("/py", None), ["/py", "-m", "commonnet.node"])

    def test_venv_python_path(self):
        self.assertTrue(str(layers.venv_python(Path("/v"))).replace("\\", "/").endswith(("bin/python", "Scripts/python.exe")))

    def test_runtime_is_the_current_interpreter_when_it_can_import_everything(self):
        calls = []
        ok = mock.Mock(returncode=0)
        python = layers.ensure_runtime("mac-arm64", source=Path("/src"), current_python="/cur", run=lambda *a, **k: calls.append(a) or ok)
        self.assertEqual(python, "/cur")
        self.assertEqual(len(calls), 1)  # one import check, nothing installed

    def test_a_missing_runtime_is_built_step_by_step_and_a_failure_stops_it(self):
        logged, commands = [], []

        def fake_run(command, **kwargs):
            commands.append(command)
            is_check = len(command) > 2 and command[1] == "-c"
            return mock.Mock(returncode=1 if is_check else 0)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(layers.LayersError) as caught:  # installs "succeed", the import check still fails
                layers.ensure_runtime("mac-arm64", source=Path(tmp), current_python="/cur", venv=Path(tmp) / "venv",
                                      run=fake_run, log=logged.append)
            self.assertIn("cannot be imported", str(caught.exception))
        joined = [" ".join(c) for c in commands]
        self.assertTrue(any("-m venv" in c for c in joined))
        self.assertTrue(any("pip install --upgrade pip" in c for c in joined))
        self.assertTrue(any("transformers==5.8.0" in c for c in joined))

        def failing(command, **kwargs):
            return mock.Mock(returncode=1)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(layers.LayersError) as caught:
                layers.ensure_runtime("mac-arm64", source=Path(tmp), current_python="/cur", venv=Path(tmp) / "venv",
                                      run=failing, log=lambda *_: None)
            self.assertIn("failed", str(caught.exception))

    def test_dry_run_installs_and_runs_nothing(self):
        def no(*a, **k):
            raise AssertionError("dry run must not run commands")

        with tempfile.TemporaryDirectory() as tmp:
            logged = []
            checks = mock.Mock(returncode=1)
            runs = []
            python = layers.ensure_runtime("mac-arm64", source=Path(tmp), current_python="/cur", venv=Path(tmp) / "venv",
                                           run=lambda c, **k: runs.append(c) or checks, dry_run=True, log=logged.append)
            self.assertTrue(all(c[1] == "-c" for c in runs))  # only the import probes ran
            self.assertTrue(any("would run" in line for line in logged))


class Source(unittest.TestCase):
    def _tarball(self, files):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def _urlopen(self, blob):
        class R:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def read(s): return blob
        return lambda url, timeout=0: R()

    def test_commonnet_src_env_wins_and_must_hold_the_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(layers.LayersError):
                layers.ensure_source({"COMMONNET_SRC": tmp})
            (Path(tmp) / "commonnet").mkdir()
            (Path(tmp) / "commonnet" / "__init__.py").write_text("")
            (Path(tmp) / "commonnet" / "node.py").write_text("")
            self.assertEqual(layers.ensure_source({"COMMONNET_SRC": tmp}), Path(tmp))

    def test_tarball_is_unpacked_to_the_commonnet_package_only(self):
        blob = self._tarball({"common-network-1/commonnet/__init__.py": b"", "common-network-1/commonnet/node.py": b"x=1",
                              "common-network-1/join/join.py": b"y=1", "common-network-1/gateway/secret.py": b"no", "common-network-1/commonnet/../../evil.py": b"no"})
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "src"
            got = layers.ensure_source({}, ref="v0.3.3", dest=dest, urlopen=self._urlopen(blob), log=lambda *_: None)
            self.assertEqual(got, dest)
            self.assertTrue((dest / "commonnet" / "node.py").is_file())
            self.assertTrue((dest / "join" / "join.py").is_file())  # the node opens its tunnel through join/
            self.assertFalse((dest / "gateway").exists())
            self.assertFalse((Path(tmp) / "evil.py").exists())
            # a pinned ref is not fetched twice
            again = layers.ensure_source({}, ref="v0.3.3", dest=dest, urlopen=lambda *a, **k: 1 / 0, log=lambda *_: None)
            self.assertEqual(again, dest)

    def test_an_archive_without_the_package_is_refused_and_leaves_nothing(self):
        blob = self._tarball({"common-network-1/README.md": b"hi"})
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "src"
            with self.assertRaises(layers.LayersError):
                layers.ensure_source({}, ref="v1", dest=dest, urlopen=self._urlopen(blob), log=lambda *_: None)
            self.assertFalse(dest.exists())

    def test_a_download_failure_is_a_clear_error(self):
        def boom(url, timeout=0):
            raise OSError("offline")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(layers.LayersError) as caught:
                layers.ensure_source({}, ref="v1", dest=Path(tmp) / "s", urlopen=boom, log=lambda *_: None)
            self.assertIn("COMMONNET_SRC", str(caught.exception))


class Run(unittest.TestCase):
    def test_run_builds_the_node_command_and_passes_the_environment(self):
        started = []

        class Child:
            def wait(self): return 7
            def send_signal(self, s): pass

        def runner(command, env=None):
            started.append((command, env))
            return Child()

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            (src / "commonnet").mkdir()
            (src / "commonnet" / "__init__.py").write_text("")
            (src / "commonnet" / "node.py").write_text("")
            ok = mock.Mock(returncode=0)
            code = layers.run("https://seed", ["--port", "9"], env={"COMMONNET_SRC": str(src)}, runner=runner,
                              run_cmd=lambda *a, **k: ok, log=lambda *_: None, nvidia=False) \
                if sys.platform == "darwin" else 7
            if sys.platform == "darwin":
                self.assertEqual(code, 7)
                command, env = started[0]
                self.assertEqual(command[1:], ["-m", "commonnet.node", "--network", "https://seed", "--port", "9"])
                self.assertTrue(env["PYTHONPATH"].startswith(str(src)))

    def test_dry_run_starts_no_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            (src / "commonnet").mkdir()
            (src / "commonnet" / "__init__.py").write_text("")
            (src / "commonnet" / "node.py").write_text("")
            lines = []
            code = layers.run("x", dry_run=True, env={"COMMONNET_SRC": str(src)}, nvidia=True,
                              runner=lambda *a, **k: 1 / 0, run_cmd=lambda *a, **k: mock.Mock(returncode=0),
                              log=lines.append)
            self.assertEqual(code, 0)
            self.assertTrue(any("would run" in l and "commonnet.node" in l for l in lines))


class JoinFlag(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run([sys.executable, str(JOIN_PY), "--no-update", *args], capture_output=True, text=True, timeout=60)

    def test_network_flag_is_refused_with_incompatible_options(self):
        for other in (["--model", "m"], ["--lan"], ["--list-catalogue"], ["--legacy"], ["--remove-permanent"]):
            with self.subTest(other=other):
                out = self._run("--network", *other)
                self.assertEqual(out.returncode, 2)
                self.assertIn("--network cannot be combined", out.stderr)

    def test_help_lists_the_option(self):
        self.assertIn("--network", self._run("--help").stdout)

    def test_api_is_only_valid_with_the_layer_network(self):
        for other in (["--model", "m"], ["--legacy"], ["--lan"], ["--remove-permanent"]):
            with self.subTest(other=other):
                out = self._run("--api", "12345", *other)
                self.assertEqual(out.returncode, 2)
                self.assertTrue("--api is only used with the layer network" in out.stderr
                                or "cannot be combined" in out.stderr, out.stderr)
        self.assertIn("--api", self._run("--help").stdout)

    def test_a_plain_join_with_api_goes_to_the_layer_network(self):
        with mock.patch.object(join, "run_network_member") as run, \
                mock.patch.object(join, "choose_mode", return_value="network"), \
                mock.patch.object(join, "announce_layer_first"), \
                mock.patch.object(sys, "argv", ["join.py", "--no-update", "--api"]):
            join.main()
        run.assert_called_once_with(None, ["--api", "11440"])

    def test_api_is_passed_to_the_node(self):
        for argv, expected in ((["--network"], []), (["--network", "--api"], ["--api", "11440"]),
                               (["--network", "https://seed", "--api", "12345"], ["--api", "12345"])):
            with self.subTest(argv=argv):
                with mock.patch.object(join, "run_network_member") as run, \
                        mock.patch.object(sys, "argv", ["join.py", "--no-update", *argv]):
                    join.main()
                run.assert_called_once_with(argv[1] if len(argv) > 1 and argv[1].startswith("http") else None, expected)

    def test_without_the_flag_layers_is_never_imported(self):
        source = JOIN_PY.read_text()
        self.assertIn("if args.network is not None:", source)
        top_level = [l for l in source.splitlines() if l.startswith(("import layers", "from layers"))]
        self.assertEqual(top_level, [])  # layers is imported lazily, inside run_network_member only
        body = source[source.index("def _load_layers"):source.index("def layer_network_supported")]
        self.assertEqual(source.count("import_module(\"layers\")"), body.count("import_module(\"layers\")"))  # only the loader imports it


class BackendImportCheck(unittest.TestCase):
    def test_a_python_without_the_backend_modules_is_not_a_runtime(self):
        seen = []

        def fake_run(command, **kw):
            seen.append(command[-1])
            return type("R", (), {"returncode": 0})()

        layers.can_import("python", None, fake_run, "mac-arm64")
        layers.can_import("python", None, fake_run, "nvidia")
        layers.can_import("python", None, fake_run)
        self.assertIn("mlx.core, mlx_lm", seen[0])
        self.assertIn("bitsandbytes", seen[1])
        self.assertNotIn("mlx", seen[2])
        self.assertNotIn("bitsandbytes", seen[2])


if __name__ == "__main__":
    unittest.main()
