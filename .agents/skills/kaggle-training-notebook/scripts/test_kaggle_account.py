"""Offline regression checks; all credential paths and network clients are isolated."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location("account_helper", Path(__file__).with_name("kaggle_account.py"))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.accounts = self.root / "accounts"
        self.accounts.mkdir()
        self.live = self.root / "credentials.json"
        self.live.write_text(json.dumps({"username": "alice", "refresh_token": "FAKE"}))
        self.mcp = self.root / ".mcp.json"
        self.mcp.write_text(json.dumps({"mcpServers": {"kaggle": {
            "headers": {"Authorization": "Bearer KGAT_alice"}}}}))
        for user in ("alice", "bob"):
            (self.accounts / f"{user}.credentials.json").write_text(
                json.dumps({"username": user, "refresh_token": "FAKE"}))
            (self.accounts / f"{user}.mcp-token").write_text(f"Bearer KGAT_{user}")
        # confirm=True is the mutating path; the gate itself is covered separately.
        self.args = argparse.Namespace(username="bob", mcp_config=self.mcp,
                                       gpu_hours=6.0, tpu_hours=0.0, confirm=True)
        helper._lock_depth = 0
        for attr, value in (("ACCOUNTS", self.accounts), ("LIVE", self.live)):
            p = patch.object(helper, attr, value)
            p.start()
            self.addCleanup(p.stop)
        self.output = io.StringIO()
        for context in (redirect_stdout(self.output), redirect_stderr(self.output)):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def q(self, user, available, reserved=0.0):
        return {"user": user, "gpu_left": available, "tpu_left": 0, "refresh_at": "",
                "gpu_reserved": reserved}

    def plan_args(self, gpu_hours, session_hours=11.0, exclude=None):
        return argparse.Namespace(gpu_hours=gpu_hours, session_hours=session_hours,
                                  exclude=exclude, mcp_config=self.mcp)

    def test_quota_subtracts_reservations_without_writing_credentials(self):
        before = {p: p.read_bytes() for p in [self.live, *self.accounts.iterdir()]}
        client = MagicMock()
        creds = MagicMock()
        creds.generate_access_token.return_value.token = "FAKE_ACCESS"
        creds.introspect.return_value = "bob"
        quota = SimpleNamespace(total_time_allowed=timedelta(hours=30),
                                time_used=timedelta(hours=10), time_reserved=timedelta(hours=16))
        response = SimpleNamespace(gpu_quota=quota, tpu_quota=None, quota_refresh_time=None)
        client.kernels.kernels_api_client.get_accelerator_quota_statistics.return_value = response
        load = MagicMock(return_value=creds)
        modules = {
            "kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=SimpleNamespace(
                build_kaggle_client_with_params=MagicMock(return_value=client))),
            "kagglesdk.kaggle_client": SimpleNamespace(KaggleClient=MagicMock(return_value=client)),
            "kagglesdk.kaggle_creds": SimpleNamespace(KaggleCredentials=SimpleNamespace(load=load)),
            "kagglesdk.kernels.types.kernels_api_service": SimpleNamespace(
                ApiGetAcceleratorQuotaStatisticsRequest=MagicMock()),
        }
        with patch.dict("sys.modules", modules):
            self.assertEqual(helper.account_quota("bob")["gpu_left"], 4)
            quota.time_reserved = timedelta(hours=25)
            self.assertEqual(helper.account_quota("bob")["gpu_left"], 0)
            creds.introspect.return_value = "wrong_owner"
            with self.assertRaises(RuntimeError):
                helper.account_quota("bob")
        self.assertEqual(load.call_args.kwargs["file_path"], str(helper.creds_path("bob")))
        creds.refresh_access_token.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_ensure_keeps_sufficient_active_and_only_switches_when_needed(self):
        quotas = {"alice": self.q("alice", 7), "bob": self.q("bob", 30)}
        with patch.object(helper, "survey", return_value=(quotas, {})), \
                patch.object(helper, "cmd_use", return_value=0) as use:
            self.assertEqual(helper.cmd_ensure(self.args), 0)
            use.assert_not_called()
            quotas["alice"]["gpu_left"] = 4
            self.assertEqual(helper.cmd_ensure(self.args), 0)
            self.assertEqual(use.call_args.args[0].username, "bob")
            use.reset_mock()
            helper.token_path("bob").unlink()
            self.assertEqual(helper.cmd_ensure(self.args), 2)
            use.assert_not_called()

    def test_unknown_quota_is_failure_without_switch(self):
        with patch.object(helper, "survey", return_value=({}, {"alice": "unavailable"})), \
                patch.object(helper, "cmd_use") as use:
            self.assertEqual(helper.cmd_ensure(self.args), 1)
            use.assert_not_called()

    def test_missing_mcp_or_mislabelled_snapshot_does_not_change_login(self):
        before = self.live.read_bytes()
        helper.token_path("bob").unlink()
        self.assertEqual(helper.cmd_use(self.args), 1)
        self.assertEqual(self.live.read_bytes(), before)
        helper.token_path("bob").write_text("KGAT_bob")
        helper.creds_path("bob").write_text('{"username": "alice"}')
        self.assertEqual(helper.cmd_use(self.args), 1)
        self.assertEqual(self.live.read_bytes(), before)

    def test_refresh_failure_is_not_success_and_leaves_mcp_unchanged(self):
        before = self.mcp.read_bytes()
        with patch.object(helper, "cmd_refresh", return_value=1), \
                patch.object(helper.subprocess, "run") as run:
            self.assertEqual(helper.cmd_use(self.args), 1)
            run.assert_not_called()
        self.assertEqual(self.mcp.read_bytes(), before)

    def test_sync_failure_is_not_success(self):
        (self.root / "scripts").mkdir()
        (self.root / "scripts/sync_kaggle_mcp.py").touch()
        with patch.object(helper, "cmd_refresh", return_value=0), \
                patch.object(helper.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
            self.assertEqual(helper.cmd_use(self.args), 1)

    def test_success_normalizes_bearer_and_private_file_mode(self):
        result = SimpleNamespace(returncode=0, stdout="username: bob\nauth_method: oauth\n")
        with patch.object(helper, "cmd_refresh", return_value=0), \
                patch.object(helper.subprocess, "run", return_value=result):
            self.assertEqual(helper.cmd_use(self.args), 0)
        self.assertEqual(helper.active_user(), "bob")
        self.assertEqual(helper.mcp_account(self.mcp), "bob")
        header = json.loads(self.mcp.read_text())["mcpServers"]["kaggle"]["headers"]["Authorization"]
        self.assertEqual(header, "Bearer KGAT_bob")
        self.assertEqual(self.mcp.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("KGAT_bob", self.output.getvalue())

    def test_survey_suppresses_exception_credentials(self):
        with patch.object(helper, "account_quota", side_effect=RuntimeError("SECRET_SENTINEL")):
            quotas, failures = helper.survey()
        helper.print_survey(quotas, failures, "alice")
        self.assertNotIn("SECRET_SENTINEL", self.output.getvalue())
        self.assertEqual(set(failures), {"alice", "bob"})

    def test_config_view_username_must_match_exactly(self):
        result = SimpleNamespace(returncode=0, stdout="username: bobby\n")
        with patch.object(helper, "cmd_refresh", return_value=0), \
                patch.object(helper.subprocess, "run", return_value=result):
            self.assertEqual(helper.cmd_use(self.args), 1)

    def test_config_view_bulleted_output_of_cli_2_2_is_accepted(self):
        # kaggle 2.2.4 prints "- username: bob"; the exact match used to fail on the bullet and
        # report a successful switch as a revoked token (exit 1) -- seen 2026-09-11.
        result = SimpleNamespace(returncode=0, stdout="Configuration values from ~/.kaggle\n"
                                                      "- username: bob\n- auth_method: OAUTH\n")
        with patch.object(helper, "cmd_refresh", return_value=0), \
                patch.object(helper.subprocess, "run", return_value=result):
            self.assertEqual(helper.cmd_use(self.args), 0)

    def test_use_without_confirm_changes_nothing(self):
        before = (self.live.read_bytes(), self.mcp.read_bytes())
        with patch.object(helper.subprocess, "run") as run, \
                patch.object(helper, "cmd_refresh") as refresh:
            self.args.confirm = False
            self.assertEqual(helper.cmd_use(self.args), 3)
            run.assert_not_called()
            refresh.assert_not_called()
        self.assertEqual((self.live.read_bytes(), self.mcp.read_bytes()), before)
        self.assertEqual(helper.active_user(), "alice")

    def test_ensure_without_confirm_proposes_and_does_not_switch(self):
        quotas = {"alice": self.q("alice", 4), "bob": self.q("bob", 30)}
        self.args.confirm = False
        with patch.object(helper, "survey", return_value=(quotas, {})), \
                patch.object(helper, "cmd_use") as use:
            self.assertEqual(helper.cmd_ensure(self.args), 3)
            use.assert_not_called()
        self.assertIn("bob", self.output.getvalue())
        self.assertEqual(helper.active_user(), "alice")

    def test_use_without_mcp_config_still_switches_the_cli(self):
        self.mcp.unlink()
        result = SimpleNamespace(returncode=0, stdout="username: bob\n")
        with patch.object(helper, "cmd_refresh", return_value=0), \
                patch.object(helper.subprocess, "run", return_value=result):
            self.assertEqual(helper.cmd_use(self.args), 0)
        self.assertEqual(helper.active_user(), "bob")

    def test_plan_splits_the_budget_and_reports_a_shortfall(self):
        quotas = {"alice": self.q("alice", 7), "bob": self.q("bob", 30, reserved=2.0)}
        with patch.object(helper, "survey", return_value=(quotas, {})):
            self.assertEqual(helper.cmd_plan(self.plan_args(20.0)), 0)
            text = self.output.getvalue()
            self.assertIn("covered 20.00 h of 20.00 h in 3 session(s) across 2 account(s)", text)
            self.assertIn("reserved", text)                      # bob may have a live session
            self.assertEqual(helper.cmd_plan(self.plan_args(40.0)), 2)
            self.assertIn("SHORTFALL 3.00 h", self.output.getvalue())

    def test_plan_honours_exclusions_and_never_switches(self):
        before = self.live.read_bytes()
        quotas = {"alice": self.q("alice", 7), "bob": self.q("bob", 30)}
        with patch.object(helper, "survey", return_value=(quotas, {})):
            self.assertEqual(helper.cmd_plan(self.plan_args(20.0, exclude=["bob"])), 2)
        self.assertIn("excluded by --exclude", self.output.getvalue())
        self.assertEqual(self.live.read_bytes(), before)

    def test_plan_skips_an_account_that_cannot_be_switched_to(self):
        helper.token_path("bob").unlink()
        quotas = {"alice": self.q("alice", 7), "bob": self.q("bob", 30)}
        with patch.object(helper, "survey", return_value=(quotas, {})):
            self.assertEqual(helper.cmd_plan(self.plan_args(20.0)), 2)
        self.assertIn("no MCP token", self.output.getvalue())

    def test_health_reports_a_dead_snapshot_without_leaking_it(self):
        args = argparse.Namespace(quiet=False, mcp_config=self.mcp)
        def identity(user):
            if user == "bob":
                raise RuntimeError("SECRET_SENTINEL")
            return user
        with patch.object(helper, "snapshot_identity", side_effect=identity):
            self.assertEqual(helper.cmd_health(args), 1)
        text = self.output.getvalue()
        self.assertIn("DEAD", text)
        self.assertNotIn("SECRET_SENTINEL", text)

    def test_health_flags_a_snapshot_the_server_assigns_to_someone_else(self):
        args = argparse.Namespace(quiet=True, mcp_config=self.mcp)
        with patch.object(helper, "snapshot_identity", return_value="alice"):
            self.assertEqual(helper.cmd_health(args), 1)
        self.assertIn("MISLABELLED", self.output.getvalue())

    def test_switch_lock_refuses_a_concurrent_holder(self):
        import fcntl
        import os
        handle = os.open(self.accounts / ".switch.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with self.assertRaises(SystemExit):
                with helper.switch_lock():
                    self.fail("the lock must not be granted twice")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)
        self.assertEqual(helper._lock_depth, 0)
        with helper.switch_lock():                       # re-entrant for use -> refresh
            with helper.switch_lock():
                self.assertEqual(helper._lock_depth, 2)
        self.assertEqual(helper._lock_depth, 0)

    def test_invalid_budgets_rejected_before_network(self):
        for value in ("-1", "nan", "inf"):
            with patch("sys.argv", ["helper", "ensure", "--gpu-hours", value]), \
                    patch.object(helper, "survey") as survey:
                with self.assertRaises(SystemExit) as caught:
                    helper.main()
                self.assertEqual(caught.exception.code, 2)
                survey.assert_not_called()


if __name__ == "__main__":
    unittest.main()
