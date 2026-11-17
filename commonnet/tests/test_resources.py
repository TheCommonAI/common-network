"""Resource budgets must pace shared work without changing inference results."""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import unittest
from unittest import mock

from commonnet import resources
from commonnet.net.protocol import Tensor
from commonnet.net.server import BlockServer, StubBlock


def settings(**overrides):
    values = dict(stage_gib=None, cpu_threads=None, compute_percent=None, bandwidth_mbps=None)
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeClock:
    def __init__(self):
        self.time = 10.0
        self.sleeps = []

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.time += seconds


class AggregatePacing(unittest.TestCase):
    def test_sequential_transfers_pay_for_all_bytes_but_idle_time_adds_no_debt(self):
        clock = FakeClock()
        pacer = resources.Pacer(100, now=clock.now, sleep=clock.sleep)
        pacer.consume(50)
        pacer.consume(150)
        self.assertEqual(clock.sleeps, [0.5, 1.5])
        clock.time += 60
        pacer.consume(100)
        self.assertEqual(clock.sleeps[-1], 1)

    def test_concurrent_callers_share_one_rate_instead_of_each_getting_the_full_rate(self):
        # Keep all sleepers parked at the same fake clock instant: reservations must serialize,
        # while sleeping must not hold the reservation lock and block subsequent callers.
        sleeps = []
        lock = threading.Lock()
        release = threading.Event()
        reserved = threading.Event()

        def sleep(seconds):
            with lock:
                sleeps.append(seconds)
                if len(sleeps) == 4:
                    reserved.set()
            release.wait(2)

        pacer = resources.Pacer(100, now=lambda: 10, sleep=sleep)
        threads = [threading.Thread(target=pacer.consume, args=(100,)) for _ in range(4)]
        try:
            for thread in threads:
                thread.start()
            self.assertTrue(reserved.wait(2), "concurrent reservations blocked behind a sleeping caller")
            self.assertEqual(sorted(sleeps), [1, 2, 3, 4])
        finally:
            release.set()
            for thread in threads:
                thread.join(2)

    def test_unlimited_or_empty_transfers_never_sleep(self):
        sleep = mock.Mock()
        for rate in (None, 0):
            resources.Pacer(rate, sleep=sleep).consume(10000)
        for amount in (0, -1):
            resources.Pacer(100, sleep=sleep).consume(amount)
        sleep.assert_not_called()


class DownloadPacing(unittest.TestCase):
    def test_read_all_is_chunked_and_every_byte_is_charged(self):
        data = bytes(range(256)) * 700
        response = io.BytesIO(data)
        original_read = response.read
        sizes = []

        def read(size=-1):
            sizes.append(size)
            return original_read(size)

        response.read = read
        budget = mock.Mock()
        with mock.patch.object(resources, "bandwidth", budget):
            with resources.PacedResponse(response) as paced:
                self.assertEqual(paced.read(), data)
        self.assertTrue(response.closed)
        self.assertTrue(all(0 < size <= 65536 for size in sizes))
        self.assertEqual(sum(call.args[0] for call in budget.consume.call_args_list), len(data))

    def test_bounded_reads_preserve_position_and_zero_reads_do_not_charge(self):
        response = io.BytesIO(b"abcdef")
        budget = mock.Mock()
        with mock.patch.object(resources, "bandwidth", budget):
            paced = resources.PacedResponse(response)
            self.assertEqual(paced.read(0), b"")
            budget.consume.assert_not_called()
            self.assertEqual(paced.read(2), b"ab")
            self.assertEqual(paced.read(20), b"cdef")
            self.assertEqual(paced.read(1), b"")
        self.assertEqual([call.args[0] for call in budget.consume.call_args_list], [2, 4])


class ResourceSettings(unittest.TestCase):
    def test_invalid_settings_are_rejected(self):
        for field in ("stage_gib", "cpu_threads", "compute_percent", "bandwidth_mbps"):
            for value in (0, -1, float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    resources.validate(settings(**{field: value}))
        for percent in (0.5, 101):
            with self.assertRaises(ValueError):
                resources.validate(settings(compute_percent=percent))

    def test_cpu_budget_sets_backend_thread_controls_and_bandwidth_uses_decimal_megabits(self):
        torch = mock.Mock()
        with mock.patch.dict(sys.modules, {"torch": torch}), mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(resources, "bandwidth"), mock.patch.object(resources, "compute_percent"):
            resources.configure(settings(cpu_threads=2, compute_percent=25, bandwidth_mbps=8))
            torch.set_num_threads.assert_called_once_with(2)
            for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                self.assertEqual(os.environ[variable], "2")
            self.assertEqual(resources.bandwidth.rate, 1_000_000)
            self.assertEqual(resources.compute_percent, 25)

    def test_unspecified_limits_restore_unlimited_pacing(self):
        with mock.patch.object(resources, "bandwidth"), mock.patch.object(resources, "compute_percent"):
            resources.configure(settings())
            self.assertIsNone(resources.bandwidth.rate)
            self.assertEqual(resources.compute_percent, 100)


class InferencePacing(unittest.TestCase):
    def test_inference_duty_cycle_inserts_the_expected_idle_share(self):
        for percent, expected in ((50, 0.2), (25, 0.6), (100, None)):
            with self.subTest(percent=percent), mock.patch.object(resources, "compute_percent", percent), \
                    mock.patch.object(resources.time, "sleep") as sleep:
                resources.idle_after(0.2)
                if expected is None:
                    sleep.assert_not_called()
                else:
                    self.assertAlmostEqual(sleep.call_args.args[0], expected)

    def test_replay_preserves_output_and_does_not_repeat_compute_or_idle(self):
        block = StubBlock(0, 4)
        server = BlockServer(block)
        tensor = Tensor((1, 1, 2), "float16", b"\x00\x00\x00\x00")
        try:
            server.open("resource-test")
            with mock.patch("commonnet.net.server.time.perf_counter", side_effect=[10, 10.2]), \
                    mock.patch.object(resources, "idle_after") as idle:
                first = server.forward("resource-test", tensor, 0)
                repeat = server.forward("resource-test", tensor, 0)
            self.assertEqual(first, tensor)
            self.assertEqual(repeat, tensor)
            self.assertEqual(block.calls.count(("forward", "resource-test", 0)), 1)
            idle.assert_called_once()
            self.assertAlmostEqual(idle.call_args.args[0], 0.2)
            self.assertAlmostEqual(server._sessions["resource-test"].stage_ms, 200)
        finally:
            server.stop()

    def test_idle_holds_the_compute_lock_so_a_second_session_cannot_use_the_reserved_share(self):
        block = StubBlock(0, 4)
        server = BlockServer(block)
        tensor = Tensor((1, 1, 2), "float16", b"\x00\x00\x00\x00")
        entered = threading.Event()
        release = threading.Event()

        def idle(_):
            entered.set()
            release.wait(2)

        try:
            server.open("first")
            server.open("second")
            with mock.patch.object(resources, "idle_after", side_effect=idle):
                first = threading.Thread(target=server.forward, args=("first", tensor, 0))
                second = threading.Thread(target=server.forward, args=("second", tensor, 0))
                first.start()
                self.assertTrue(entered.wait(2))
                second.start()
                self.assertFalse(server._lock.acquire(blocking=False))
                self.assertNotIn(("forward", "second", 0), block.calls)
                release.set()
                first.join(2)
                second.join(2)
                self.assertFalse(first.is_alive())
                self.assertFalse(second.is_alive())
        finally:
            release.set()
            server.stop()


if __name__ == "__main__":
    unittest.main()
