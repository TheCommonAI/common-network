"""coordinator --network: argument rules and a clean error when nothing answers."""
import contextlib
import io
import unittest

from commonnet import coordinator


def run(*argv):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = coordinator.main(list(argv))
    return code, err.getvalue()


class CoordinatorNetworkCli(unittest.TestCase):
    def test_exactly_one_of_endpoints_and_network(self):
        code, err = run("--prompt", "hi")
        self.assertEqual(code, 2)
        self.assertIn("exactly one", err)
        code, err = run("--prompt", "hi", "--endpoints", "http://x@0:4", "--network", "http://y")
        self.assertEqual(code, 2)
        self.assertIn("exactly one", err)

    def test_a_network_nobody_answers_is_a_one_line_error_not_a_traceback(self):
        code, err = run("--prompt", "hi", "--network", "http://127.0.0.1:1")
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertIn("seed", err)
        self.assertNotIn("Traceback", err)

    def test_the_network_client_is_mlx4_with_float32_hidden_states(self):
        import torch
        from unittest import mock

        seen = {}

        class SpyClient:
            def __init__(self, checkpoint, transport, **kw):
                seen.update(kw, transport=transport)

            def generate(self, endpoints, ids, n, **sampling):
                return [1, 2, 3]

        class FakeNetwork:
            def plan(self):
                return ["endpoints"], "transport"

        with mock.patch("commonnet.client4.Client4", SpyClient):
            self.assertEqual(coordinator.generate_network(FakeNetwork(), "ckpt", [5], 3), [1, 2, 3])
        self.assertEqual(seen["fmt"], "mlx4")
        self.assertEqual(seen["hidden_dtype"], torch.float32)
        self.assertEqual(seen["transport"], "transport")

if __name__ == "__main__":
    unittest.main()
