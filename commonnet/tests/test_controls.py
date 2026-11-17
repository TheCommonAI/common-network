import http.client
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
from urllib.parse import urlsplit

from commonnet import controls


class ControlCommands(unittest.TestCase):
    def test_apply_preserves_network_and_api_choices_without_duplicate_limits(self):
        config = {"argv": ["/python", "/join.py", "--network=model", "--no-api", "--stage-gib", "8",
                           "--cpu-threads=3", "--compute-percent=75", "--bandwidth-mbps", "20", "--permanent"]}
        limits = dict(stage_gib=4, cpu_threads=2, compute_percent=50, bandwidth_mbps=None)
        with mock.patch.object(controls.os, "cpu_count", return_value=8):
            command = controls.apply_command(config, limits)
        self.assertEqual(command, ["/python", "/join.py", "--network=model", "--no-api", "--stage-gib=4",
                                   "--cpu-threads=2", "--compute-percent=50", "--permanent"])

    def test_apply_reinstalls_foreground_service_command_without_conflicting_mode(self):
        config = {"argv": ["python", "join.py", "--foreground", "--network=qwen3-8b", "--no-api",
                           "--stage-gib=4", "--no-update"]}
        command = controls.apply_command(config, {"stage_gib": 6})
        self.assertEqual(command, ["python", "join.py", "--network=qwen3-8b", "--no-api", "--no-update",
                                   "--stage-gib=6", "--permanent"])
        self.assertNotIn("--foreground", command)

    def test_automatic_removes_alias_and_restores_runtime_defaults(self):
        command = controls.apply_command({"argv": ["p", "j", "--memory-gib=4", "--network"]}, {})
        self.assertEqual(command, ["p", "j", "--network", "--permanent"])

    def test_invalid_values_fail_before_service_mutation(self):
        invalid = [{"stage_gib": float("nan")}, {"bandwidth_mbps": -1}, {"compute_percent": 101},
                   {"compute_percent": 0.5}, {"cpu_threads": 1.5}, {"cpu_threads": True}, {"unknown": 4}]
        for limits in invalid:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                controls.validate_limits(limits)

    def test_only_one_companion_can_hold_a_config_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "service.json"
            first = controls.acquire_instance(path)
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(controls.acquire_instance(path))
            finally:
                first.close()
            second = controls.acquire_instance(path)
            self.assertIsNotNone(second)
            second.close()

    def test_service_exit_errors_are_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "service.json"
            path.write_text(json.dumps({"argv": ["p", "j", "--network"]}))
            run = mock.Mock(return_value=subprocess.CompletedProcess([], 1, "", "specific error"))
            with self.assertRaisesRegex(RuntimeError, "specific error"):
                controls.change_service(path, {}, run=run)
            controls.change_service(path, stop=True, run=mock.Mock(return_value=subprocess.CompletedProcess([], 0)))
            self.assertEqual(run.call_args.args[0], ["p", "j", "--network", "--permanent"])


class LiveNodeStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "status.json"

    def tearDown(self):
        self.tmp.cleanup()

    def status(self, *, alive=True, **changes):
        data = {"state": "up", "pid": 123, "ts": 100, "window": [0, 16],
                "stages": [[0, 16], [16, 32]], "coverage": [1, 0]}
        data.update(changes)
        self.path.write_text(json.dumps(data))
        return controls.node_status(self.path, now=101, pid_alive=lambda pid: alive)

    def test_online_host_does_not_claim_complete_requests_or_useful_work(self):
        status = self.status()
        self.assertEqual(status["state"], "waiting")
        self.assertIn("hosting layers 0–16", status["title"])
        self.assertIn("Waiting for peers", status["detail"])
        self.assertEqual(status["activity"], "")
        self.assertEqual(self.status(coverage=[1, 1])["state"], "online")
        self.assertEqual(self.status(coverage=[])["state"], "waiting")

    def test_dead_pid_and_stale_documents_never_show_online(self):
        self.assertEqual(self.status(alive=False)["state"], "stopped")
        self.assertEqual(self.status(ts=80)["state"], "stale")
        self.assertEqual(self.status(ts=float("nan"))["state"], "stale")
        self.assertEqual(self.status(state="stopped")["state"], "stopped")
        self.path.write_text("{")
        self.assertEqual(controls.node_status(self.path)["state"], "stopped")

    def test_preparing_and_actionable_error_states(self):
        for state in ("starting", "loading", "moving"):
            self.assertEqual(self.status(state=state)["state"], state)
        status = self.status(state="downloading", download={"bytes": 50, "total": 100})
        self.assertIn("50%", status["detail"])
        status = self.status(state="error", error="Not enough memory for these layers. Lower the model size.")
        self.assertEqual(status["state"], "error")
        self.assertIn("Not enough memory", status["detail"])
        self.assertNotIn("Traceback", controls.actionable_error("Traceback (most recent call last): returned non-zero exit status 5"))

    def test_work_counters_are_labelled_model_sessions_and_include_local_work(self):
        status = self.status(activity={"stage_sessions": 4, "window_seconds": 300, "completed_requests": 2})
        self.assertIn("4 model sessions processed", status["activity"])
        self.assertIn("includes local work", status["activity"])
        self.assertNotIn("requests served", status["activity"])
        self.assertIn("No model sessions processed", self.status(activity={"stage_sessions": 0})["activity"])


class ControlPanelHTTP(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "service.json"
        self.path.write_text(json.dumps({"argv": ["p", "j", "--network"], "limits": {"stage_gib": 4}}))
        self.server, self.url = controls.make_server(self.path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = urlsplit(self.url).path.rstrip("/")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        status, raw = response.status, response.read()
        conn.close()
        return status, raw

    def test_panel_and_saved_config_are_available_only_on_secret_path(self):
        status, raw = self.request("GET", self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Apply limits", raw)
        status, raw = self.request("GET", self.base + "/config")
        self.assertEqual(json.loads(raw)["limits"]["stage_gib"], 4)
        self.assertEqual(self.request("GET", "/")[0], 404)
        self.assertEqual(self.request("GET", self.base + "/", headers={"Host": "evil.example"})[0], 403)

    def test_live_status_endpoint_uses_the_same_summary_as_native(self):
        with mock.patch.object(controls, "node_status", return_value={"state": "waiting", "title": "Online", "detail": "Waiting for peers", "activity": ""}):
            status, raw = self.request("GET", self.base + "/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["state"], "waiting")

    def test_apply_validates_body_and_invokes_service(self):
        with mock.patch.object(controls, "change_service") as change:
            status, raw = self.request("POST", self.base + "/apply", json.dumps({"compute_percent": 50}))
            self.assertEqual(status, 200)
            change.assert_called_once_with(self.path, {"stage_gib": None, "cpu_threads": None, "compute_percent": 50, "bandwidth_mbps": None}, stop=False)
            status, raw = self.request("POST", self.base + "/apply", "{" )
            self.assertEqual(status, 400)
            self.assertEqual(change.call_count, 1)


if __name__ == "__main__":
    unittest.main()
