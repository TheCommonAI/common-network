"""Background nodes keep donor limits and recover using native service managers."""
import argparse
import contextlib
import io
import json
import re
import shlex
import plistlib
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import join

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def service_args(**changes):
    values = dict(network="qwen3-8b", no_update=True, api=None, no_api=True,
                  cpu_threads=2, compute_percent=35.0, bandwidth_mbps=8.0,
                  stage_gib=4.0, layers=None, local_front="off", no_draft=True,
                  direct=False, no_direct=True)
    values.update(changes)
    return argparse.Namespace(**values)


class SavedResourceChoices(unittest.TestCase):
    def test_resource_limits_and_no_api_survive_a_service_restart(self):
        argv = join._service_argv(service_args())
        self.assertIn("--no-api", argv)
        self.assertNotIn("--permanent", argv)
        with mock.patch.object(sys, "argv", argv[1:]), \
                mock.patch.object(join, "run_network_member") as run, \
                mock.patch.object(join, "print"):
            join.main()
        network, flags = run.call_args.args
        self.assertEqual(network, "qwen3-8b")
        for flag in ("--cpu-threads=2", "--compute-percent=35.0", "--bandwidth-mbps=8.0"):
            self.assertIn(flag, flags)
        self.assertEqual(flags[flags.index("--stage-gib") + 1], "4.0")
        self.assertNotIn("--api", flags)
        self.assertIn("--no-rtc", flags)
        self.assertIn("--no-draft", flags)

    def test_memory_alias_reaches_the_node(self):
        argv = ["join.py", "--no-update", "--foreground", "--network=qwen3-8b", "--memory-gib=3.5"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(join, "run_network_member") as run, \
                mock.patch.object(join, "print"):
            join.main()
        flags = run.call_args.args[1]
        self.assertEqual(flags[flags.index("--stage-gib") + 1], "3.5")

    def test_invalid_limits_are_rejected_before_setup_or_update(self):
        bad = ("--memory-gib=0", "--memory-gib=nan", "--stage-gib=inf", "--stage-gib=-1",
               "--cpu-threads=0", "--cpu-threads=1.5", "--compute-percent=0.5",
               "--compute-percent=101", "--compute-percent=nan", "--bandwidth-mbps=-1",
               "--bandwidth-mbps=inf")
        for flag in bad:
            with self.subTest(flag=flag), \
                    mock.patch.object(sys, "argv", ["join.py", flag]), \
                    mock.patch.object(join, "_load_layers") as layers, \
                    mock.patch.object(join, "fetch_update") as update, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    join.main()
                self.assertEqual(error.exception.code, 2)
                layers.assert_not_called()
                update.assert_not_called()


class DirectLauncherCompatibility(unittest.TestCase):
    def test_old_direct_launcher_command_stays_foreground_without_reinstalling(self):
        with mock.patch.object(sys, "argv", ["join.py", "--no-update"]), \
                mock.patch.object(join, "choose_mode", return_value="network"), \
                mock.patch.object(join, "install_permanent") as install, \
                mock.patch.object(join, "run_network_member") as foreground, \
                mock.patch.object(join, "print"):
            join.main()
        foreground.assert_called_once_with(None, ["--api", "11440", "--auto-direct"], debug=False)
        install.assert_not_called()

    def test_foreground_opts_out_of_service_and_background_controls(self):
        with mock.patch.object(sys, "argv", ["join.py", "--no-update", "--foreground"]), \
                mock.patch.object(join, "choose_mode", return_value="network"), \
                mock.patch.object(join, "install_permanent") as install, \
                mock.patch.object(join, "run_network_member") as foreground, \
                mock.patch.object(join, "print"):
            join.main()
        foreground.assert_called_once_with(None, ["--api", "11440", "--auto-direct"], debug=False)
        install.assert_not_called()

    def test_service_command_runs_foreground_without_reinstalling_itself(self):
        argv = join._service_argv(service_args(no_controls=True))
        self.assertIn("--foreground", argv)
        self.assertIn("--no-controls", argv)
        with mock.patch.object(sys, "argv", argv[1:]), \
                mock.patch.object(join, "install_permanent") as install, \
                mock.patch.object(join, "run_network_member") as foreground, \
                mock.patch.object(join, "print"):
            join.main()
        foreground.assert_called_once()
        install.assert_not_called()

    def test_foreground_and_permanent_are_mutually_exclusive_before_setup(self):
        with mock.patch.object(sys, "argv", ["join.py", "--foreground", "--permanent"]), \
                mock.patch.object(join, "_load_layers") as layers, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                join.main()
        self.assertEqual(error.exception.code, 2)
        layers.assert_not_called()

    def test_no_controls_is_kept_by_explicit_background_install(self):
        with mock.patch.object(sys, "argv", ["join.py", "--no-update", "--permanent", "--no-controls"]), \
                mock.patch.object(join, "choose_mode", return_value="network"), \
                mock.patch.object(join, "install_permanent") as install, \
                mock.patch.object(join, "print"):
            join.main()
        self.assertTrue(install.call_args.args[0].no_controls)

    def test_check_only_never_installs_a_service(self):
        layers = mock.Mock()
        layers.report.return_value = 0
        with mock.patch.object(sys, "argv", ["join.py", "--check"]), \
                mock.patch.object(join, "_load_layers", return_value=layers), \
                mock.patch.object(join, "install_permanent") as install:
            with self.assertRaises(SystemExit) as error:
                join.main()
        self.assertEqual(error.exception.code, 0)
        install.assert_not_called()

    def test_remove_only_does_not_install_the_background_default(self):
        with mock.patch.object(sys, "argv", ["join.py", "--no-update", "--remove-permanent"]), \
                mock.patch.object(join, "remove_permanent") as remove, \
                mock.patch.object(join, "install_permanent") as install, \
                mock.patch.object(join, "print"):
            join.main()
        remove.assert_called_once()
        install.assert_not_called()

    def test_controls_only_opens_companion_without_reinstalling_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "service.json"
            config.write_text("{}")
            with mock.patch.object(sys, "argv", ["join.py", "--controls"]), \
                    mock.patch.object(join, "service_config_path", return_value=config), \
                    mock.patch.object(join, "_controls_command", return_value=["python", "controls.py"]), \
                    mock.patch.object(join.subprocess, "Popen") as opened, \
                    mock.patch.object(join, "install_permanent") as install:
                join.main()
        self.assertEqual(opened.call_args.args[0], ["python", "controls.py"])
        install.assert_not_called()


class NativeServiceRecovery(unittest.TestCase):
    def test_macos_recovery_and_argv_are_written_without_a_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            plist = Path(tmp) / "node.plist"
            argv = ["/path with spaces/python", "/path with spaces/join.py", "--network=x&y"]
            with mock.patch.object(join, "_macos_plist_path", return_value=plist), \
                    mock.patch.object(Path, "home", return_value=Path(tmp)), \
                    mock.patch.object(join.subprocess, "run") as run, \
                    mock.patch.object(join, "print"):
                join.install_macos_service(argv)
            settings = plistlib.loads(plist.read_bytes())
            self.assertEqual(settings["ProgramArguments"], argv)
            self.assertTrue(settings["RunAtLoad"])
            self.assertTrue(settings["KeepAlive"])
            self.assertEqual(settings["ThrottleInterval"], 5)
            self.assertEqual(settings["StandardOutPath"], settings["StandardErrorPath"])
            self.assertEqual(run.call_args_list[-1].args[0][1], "bootstrap")

    def test_linux_retries_outlast_start_rate_limit_and_reinstall_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "node.service"
            with mock.patch.object(join, "_systemd_unit_path", return_value=unit), \
                    mock.patch.object(join.subprocess, "run") as run, \
                    mock.patch.dict(join.os.environ, {"PATH": "/path with spaces/100%/bin"}), \
                    mock.patch.object(join, "print"):
                join.install_linux_service(["/python path", "/join.py", "--network=https://host/x%20y?$a"])
            text = unit.read_text()
            self.assertIn("StartLimitIntervalSec=0", text)
            self.assertIn("Restart=always", text)
            self.assertIn("RestartSec=5", text)
            self.assertIn('Environment="PATH=/path with spaces/100%%/bin"', text)
            self.assertIn('ExecStart="/python path" "/join.py" "--network=https://host/x%%20y?$$a"', text)
            self.assertEqual(run.call_args_list[-1].args[0],
                             ["systemctl", "--user", "restart", join.SYSTEMD_UNIT])

    def test_systemd_quotes_literal_special_characters(self):
        self.assertEqual(join._systemd_quote('a"b\\c\nx%$y', command=True),
                         '"a\\"b\\\\c\\nx%%$$y"')

    def test_windows_task_restarts_without_time_limit_or_elevation(self):
        argv = [r"C:\Users\Donor Name\Python\python.exe", r"C:\Common\join.py", "--no-api"]
        root = ET.fromstring(join._windows_task_xml(argv, Path("join.log"), r"DOMAIN\Donor & Co"))
        find = lambda path: root.findtext(path, namespaces=NS)
        self.assertEqual(find("t:Settings/t:RestartOnFailure/t:Interval"), "PT1M")
        self.assertEqual(find("t:Settings/t:RestartOnFailure/t:Count"), "999")
        self.assertEqual(find("t:Settings/t:ExecutionTimeLimit"), "PT0S")
        self.assertEqual(find("t:Settings/t:MultipleInstancesPolicy"), "IgnoreNew")
        self.assertEqual(find("t:Principals/t:Principal/t:RunLevel"), "LeastPrivilege")
        self.assertEqual(find("t:Principals/t:Principal/t:LogonType"), "InteractiveToken")
        self.assertEqual(find("t:Triggers/t:LogonTrigger/t:UserId"), r"DOMAIN\Donor & Co")
        self.assertEqual(find("t:Actions/t:Exec/t:Command"), r"C:\Users\Donor Name\Python\pythonw.exe")

    def test_windows_wrapper_preserves_argv_logs_and_nonzero_exit(self):
        argv = [r"C:\path with spaces\python.exe", r"C:\Common\join.py",
                '--network=https://host/x?x="quoted"&y=%PATH%', "--cpu-threads=2"]
        with mock.patch.object(join.subprocess, "list2cmdline", return_value="captured") as encode:
            join._windows_task_xml(argv, Path("join log.txt"), "donor")
        command = encode.call_args.args[0]
        self.assertEqual(command[0], "-c")
        with mock.patch("builtins.open", mock.mock_open()) as opened, \
                mock.patch.object(subprocess, "call", return_value=17) as call:
            with self.assertRaises(SystemExit) as error:
                exec(command[1], {})
        self.assertEqual(error.exception.code, 17)
        self.assertEqual(call.call_args.args[0], argv)
        self.assertNotIn("shell", call.call_args.kwargs)
        self.assertEqual(call.call_args.kwargs["stderr"], subprocess.STDOUT)
        opened.assert_called_once_with("join log.txt", "ab", buffering=0)

    def test_windows_replacing_task_stops_old_process_before_start(self):
        written = []
        def command(argv, **kwargs):
            if argv[:2] == ["schtasks", "/create"]:
                written.append(ET.fromstring(Path(argv[argv.index("/xml") + 1]).read_bytes()))
            return subprocess.CompletedProcess(argv, 0, stdout="DOMAIN\\donor\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(Path, "home", return_value=Path(tmp)), \
                mock.patch.object(join.subprocess, "run", side_effect=command) as run, \
                mock.patch.object(join, "print"):
            join.install_windows_service(["python.exe", "join.py", "--network"])
        self.assertEqual(len(written), 1)
        actions = [c.args[0][:2] for c in run.call_args_list]
        self.assertEqual(actions, [["whoami"], ["schtasks", "/end"],
                                   ["schtasks", "/create"], ["schtasks", "/run"]])


class CompanionIntegration(unittest.TestCase):
    def test_controls_command_uses_selected_source_runtime_and_explicit_config(self):
        layers = mock.Mock()
        layers.ensure_source.return_value = Path("/local/check out")
        layers.platform_kind.return_value = "mac-arm64"
        layers.ensure_runtime.return_value = Path("/private/python")
        layers._source_prefix.return_value = "SOURCE_PREFIX; "
        with mock.patch.object(join, "_load_layers", return_value=layers), \
                mock.patch.object(join.platform, "system", return_value="Darwin"), \
                mock.patch.object(join.platform, "mac_ver", return_value=("26.3", (), "")), \
                mock.patch.object(join, "service_config_path", return_value=Path("/private/service.json")), \
                mock.patch.object(join.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            command = join._controls_command()
        layers.ensure_runtime.assert_called_once_with("mac-arm64", source=Path("/local/check out"))
        self.assertEqual(command[:2], ["/private/python", "-c"])
        self.assertTrue(command[2].startswith("SOURCE_PREFIX; "))
        self.assertIn("commonnet.controls", command[2])
        self.assertEqual(command[-2:], ["--config", "/private/service.json"])
        run.assert_not_called()  # modern macOS uses the packaged SwiftUI app, without PyObjC/pip

    def test_successful_install_persists_exact_argv_and_limits_before_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "service.json"
            def install_ui():
                saved = json.loads(path.read_text())
                self.assertEqual(saved["argv"], join._service_argv(service_args()))
                self.assertEqual(saved["limits"], dict(stage_gib=4.0, cpu_threads=2,
                                                      compute_percent=35.0, bandwidth_mbps=8.0))
            with mock.patch.object(join.platform, "system", return_value="Darwin"), \
                    mock.patch.object(join, "install_macos_service") as install, \
                    mock.patch.object(join, "service_config_path", return_value=path), \
                    mock.patch.object(join, "install_controls", side_effect=install_ui) as ui:
                join.install_permanent(service_args())
            install.assert_called_once()
            ui.assert_called_once()
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_failed_service_install_preserves_previous_config_and_skips_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "service.json"
            path.write_text("previous config")
            with mock.patch.object(join.platform, "system", return_value="Darwin"), \
                    mock.patch.object(join, "install_macos_service", side_effect=OSError("not installed")), \
                    mock.patch.object(join, "service_config_path", return_value=path), \
                    mock.patch.object(join, "install_controls") as ui:
                with self.assertRaises(OSError):
                    join.install_permanent(service_args())
            self.assertEqual(path.read_text(), "previous config")
            ui.assert_not_called()

    def test_no_controls_still_saves_resource_choices_without_bootstrapping_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "service.json"
            with mock.patch.object(join.platform, "system", return_value="Linux"), \
                    mock.patch.object(join, "install_linux_service"), \
                    mock.patch.object(join, "service_config_path", return_value=path), \
                    mock.patch.object(join, "install_controls") as ui:
                join.install_permanent(service_args(no_controls=True))
            self.assertIn("--no-controls", json.loads(path.read_text())["argv"])
            ui.assert_not_called()

    def test_ui_setup_failure_leaves_successful_node_install_and_config(self):
        for error in (RuntimeError("missing UI"), OSError("missing executable"),
                      subprocess.CalledProcessError(1, ["pip"])):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "service.json"
                with mock.patch.object(join.platform, "system", return_value="Darwin"), \
                        mock.patch.object(join, "install_macos_service"), \
                        mock.patch.object(join, "service_config_path", return_value=path), \
                        mock.patch.object(join, "install_controls", side_effect=error), \
                        mock.patch.object(join, "print") as printed:
                    join.install_permanent(service_args())
                self.assertEqual(json.loads(path.read_text())["limits"]["cpu_threads"], 2)
                self.assertTrue(any("tray setup failed" in str(c) for c in printed.call_args_list))

    def test_linux_headless_install_skips_ui_dependencies(self):
        with mock.patch.object(join.platform, "system", return_value="Linux"), \
                mock.patch.dict(join.os.environ, {}, clear=True), \
                mock.patch.object(join, "_controls_command") as command, \
                mock.patch.object(join, "print"):
            join.install_controls()
        command.assert_not_called()

    def test_applying_limits_does_not_replace_its_own_companion_service(self):
        for system in ("Darwin", "Linux", "Windows"):
            with self.subTest(system=system), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                mac = home / "Library/LaunchAgents" / (join.CONTROLS_LABEL + ".plist")
                linux = home / ".config/autostart/common-controls.desktop"
                for path in (mac, linux):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("already installed")
                with mock.patch.object(join.platform, "system", return_value=system), \
                        mock.patch.object(Path, "home", return_value=home), \
                        mock.patch.dict(join.os.environ, {"DISPLAY": ":1"}), \
                        mock.patch.object(join, "_controls_command") as command, \
                        mock.patch.object(join.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
                    join.install_controls()
                command.assert_not_called()
                if system == "Windows":
                    run.assert_called_once_with(["schtasks", "/query", "/tn", join.CONTROLS_TASK], capture_output=True)
                else:
                    run.assert_not_called()

    def test_linux_autostart_preserves_literal_arguments_through_both_escape_passes(self):
        command = ["/path with spaces/python", "-c", 'print("$donor`100%")', "back\\slash", "line\nfeed"]
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with mock.patch.object(join.platform, "system", return_value="Linux"), \
                    mock.patch.object(Path, "home", return_value=home), \
                    mock.patch.dict(join.os.environ, {"DISPLAY": ":1"}), \
                    mock.patch.object(join, "_controls_command", return_value=command), \
                    mock.patch.object(join.subprocess, "Popen") as started:
                join.install_controls()
            desktop = (home / ".config/autostart/common-controls.desktop").read_text()
            encoded = next(line[5:] for line in desktop.splitlines() if line.startswith("Exec="))
            escapes = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}
            decoded = re.sub(r"\\([nrt\\])", lambda m: escapes[m[1]], encoded)
            actual = [value.replace("%%", "%").replace("\\$", "$").replace("\\`", "`")
                      for value in shlex.split(decoded)]
            self.assertEqual(actual, command)
            self.assertEqual(started.call_args.args[0], command)

    def test_removal_deletes_saved_limits_and_linux_autostart(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = home / ".common-network/service.json"
            desktop = home / ".config/autostart/common-controls.desktop"
            for f in (path, desktop):
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("old")
            with mock.patch.object(join.platform, "system", return_value="Linux"), \
                    mock.patch.object(Path, "home", return_value=home), \
                    mock.patch.object(join, "remove_linux_service") as remove:
                join.remove_permanent()
            remove.assert_called_once()
            self.assertFalse(path.exists())
            self.assertFalse(desktop.exists())


if __name__ == "__main__":
    unittest.main()


class MacReloadRetry(unittest.TestCase):
    def test_bootstrap_retries_teardown_race(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(join, "_macos_plist_path", return_value=Path(tmp)/"node.plist"), mock.patch.object(join.Path, "home", return_value=Path(tmp)), mock.patch.object(join.time, "sleep") as sleep, mock.patch.object(join.subprocess, "run") as run:
            run.side_effect = [subprocess.CompletedProcess([], 0), subprocess.CalledProcessError(5, "launchctl", stderr="Bootstrap failed: 5"), subprocess.CompletedProcess([], 0)]
            join.install_macos_service(["python", "join.py"])
            self.assertEqual(run.call_count, 3)
            sleep.assert_called_once_with(0.5)
