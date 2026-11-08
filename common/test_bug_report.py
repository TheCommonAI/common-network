"""Offline checks for the CLI's opt-in GitHub issue drafts."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


class Terminal(io.StringIO):
    def isatty(self):
        return True


class BugReportTests(unittest.TestCase):
    def test_issue_contains_error_and_redacts_credentials(self):
        url = common._bug_report_url(
            f"{Path.home()}/test.py: failed & broken\n"
            "https://user:pass@example.com/api?token=private\n"
            "Authorization: Bearer private\napi_key=private")
        parts = urlsplit(url)
        fields = parse_qs(parts.query)
        self.assertEqual(parts.netloc, "github.com")
        self.assertEqual(parts.path, "/TheCommonAI/common-network/issues/new")
        self.assertIn("failed & broken", fields["body"][0])
        self.assertIn(common.VERSION, fields["body"][0])
        self.assertNotIn("private", fields["body"][0].split("## Environment")[0])
        self.assertNotIn("user:pass", url)
        self.assertNotIn(str(Path.home()), fields["body"][0])
        self.assertLess(len(common._bug_report_url("💥" * 10000)), 8000)

    def test_prompt_accept_decline_and_eof(self):
        for answer, opened in [("y", True), ("YES", True), ("n", False), ("", False)]:
            with self.subTest(answer=answer), patch.object(sys, "stdin", Terminal()), \
                    patch.object(sys, "stderr", Terminal()), \
                    patch("builtins.input", return_value=answer), \
                    patch.object(common.webbrowser, "open", return_value=True) as browser:
                common._offer_bug_report("RuntimeError: broken")
                self.assertEqual(browser.called, opened)
        with patch.object(sys, "stdin", Terminal()), patch.object(sys, "stderr", Terminal()), \
                patch("builtins.input", side_effect=EOFError):
            common._offer_bug_report("broken")

    def test_noninteractive_does_not_prompt(self):
        with patch.object(sys, "stdin", io.StringIO()), patch("builtins.input") as prompt:
            common._offer_bug_report("broken")
            prompt.assert_not_called()

    def test_crash_and_handled_failure_preserve_exit_and_stderr(self):
        def handled():
            print("gateway failed", file=sys.stderr)
            raise SystemExit(1)

        for action, code, detail in [(handled, 1, "gateway failed"),
                                     (lambda: 1 / 0, 1, "ZeroDivisionError")]:
            output = Terminal()
            with patch.object(sys, "stderr", output), patch.object(common, "main", action), \
                    patch.object(common, "_offer_bug_report") as report:
                with self.assertRaises(SystemExit) as raised:
                    common.cli()
                self.assertEqual(raised.exception.code, code)
                self.assertIn(detail, report.call_args.args[0])
                self.assertIs(sys.stderr, output)
                self.assertIn(detail, output.getvalue())

    def test_usage_success_and_interrupt_do_not_offer(self):
        for exception in [SystemExit(0), SystemExit(2), KeyboardInterrupt()]:
            with patch.object(common, "main", side_effect=exception), \
                    patch.object(common, "_offer_bug_report") as report:
                with self.assertRaises(SystemExit):
                    common.cli()
                report.assert_not_called()

    def test_browser_failure_prints_link(self):
        output = Terminal()
        with patch.object(sys, "stdin", Terminal()), patch.object(sys, "stderr", output), \
                patch("builtins.input", return_value="y"), \
                patch.object(common.webbrowser, "open", side_effect=OSError):
            common._offer_bug_report("broken")
        self.assertIn("https://github.com/TheCommonAI/common-network/issues/new?", output.getvalue())


if __name__ == "__main__":
    unittest.main()
