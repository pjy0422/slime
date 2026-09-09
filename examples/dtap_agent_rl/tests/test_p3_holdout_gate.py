import subprocess
import sys

import pytest

from examples.dtap_agent_rl.scripts import run_p3_holdout_gate


def test_holdout_gate_retries_only_failed_matrix_with_lower_concurrency(
    monkeypatch, tmp_path,
):
    calls = []
    returncodes = iter((1, 1, 0))

    def run(command, **kwargs):
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, next(returncodes))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(run_p3_holdout_gate.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", [
        "run_p3_holdout_gate",
        "--dtap-root", str(tmp_path / "dtap"),
        "--slime-root", str(tmp_path / "slime"),
        "--artifacts-root", str(tmp_path / "artifacts"),
        "--skip-overlay",
    ])

    with pytest.raises(SystemExit) as stopped:
        run_p3_holdout_gate.main()

    assert stopped.value.code == 0
    assert len(calls) == 3
    commands = [call[0] for call in calls]
    assert [
        command[command.index("--max-parallel") + 1] for command in commands
    ] == ["8", "4", "2"]
    assert "--resume" not in commands[0]
    assert "--resume" in commands[1]
    assert "--resume" in commands[2]
    assert all("--selection-profile" in command for command in commands)
    assert all("holdout-v1" in command for command in commands)


def test_holdout_gate_stops_after_a_green_first_pass(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [
        "run_p3_holdout_gate",
        "--dtap-root", str(tmp_path / "dtap"),
        "--slime-root", str(tmp_path / "slime"),
        "--artifacts-root", str(tmp_path / "artifacts"),
        "--skip-overlay",
    ])

    with pytest.raises(SystemExit) as stopped:
        run_p3_holdout_gate.main()

    assert stopped.value.code == 0
    assert len(calls) == 1
