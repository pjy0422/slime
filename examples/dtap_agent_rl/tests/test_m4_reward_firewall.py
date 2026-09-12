import os
from pathlib import Path

import pytest
from examples.dtap_agent_rl.reward_firewall import JudgeVerdictReader, VerdictError, find_single_judge_result
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy


def _reader(**kwargs):
    return JudgeVerdictReader(M4SecurityPolicy(max_submit_calls=2, **kwargs))


def test_only_boolean_attack_success_crosses_reward_firewall(tmp_path: Path):
    path = tmp_path / "judge_result.json"
    path.write_text(
        '{"attack_success":false,"task_success":true,"rationale":"SECRET"}',
        encoding="utf-8",
    )
    verdict = _reader().read(path, result_root=tmp_path)
    assert verdict.attack_success is False
    assert tuple(verdict.__dict__) == ("attack_success",)


@pytest.mark.parametrize(
    "body",
    [
        '{"attack_success":"false"}',
        '{"attack_success":0}',
        '{"attack_success":null}',
        '{"attack_success":false,"attack_success":true}',
        "[]",
    ],
)
def test_malformed_or_ambiguous_verdict_is_rejected(tmp_path: Path, body: str):
    path = tmp_path / "judge_result.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(VerdictError):
        _reader().read(path, result_root=tmp_path)


def test_oversized_and_symlinked_verdicts_are_rejected(tmp_path: Path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"attack_success":true,"padding":"xxxxxxxx"}')
    link = tmp_path / "judge_result.json"
    link.symlink_to(outside)
    with pytest.raises(VerdictError):
        _reader(max_judge_bytes=16).read(link, result_root=tmp_path)


def test_single_judge_discovery_rejects_ambiguity(tmp_path: Path):
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "judge_result.json").write_text('{"attack_success":false}')
    with pytest.raises(VerdictError):
        find_single_judge_result(tmp_path)


def test_verdict_reader_rejects_unexpected_file_owner(tmp_path: Path):
    path = tmp_path / "judge_result.json"
    path.write_text('{"attack_success":true}')
    reader = JudgeVerdictReader(
        M4SecurityPolicy(max_submit_calls=1),
        expected_uid=os.getuid() + 1,
    )
    with pytest.raises(VerdictError, match="ownership"):
        reader.read(path, result_root=tmp_path)
