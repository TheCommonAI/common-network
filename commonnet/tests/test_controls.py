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
