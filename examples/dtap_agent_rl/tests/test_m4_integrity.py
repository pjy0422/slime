import os
from pathlib import Path

import pytest
from examples.dtap_agent_rl.integrity import BenchmarkIntegrityGuard, IntegrityError, safe_copy_tree


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    task.mkdir()
    (task / "config.yaml").write_text("Task: {}\n", encoding="utf-8")
    (task / "judge.py").write_text("# judge\n", encoding="utf-8")
    assets = task / "assets"
    assets.mkdir()
    (assets / "message.txt").write_text("hello", encoding="utf-8")
    return task


@pytest.mark.parametrize("mutation", ["content", "add", "remove", "mode"])
def test_whole_task_manifest_detects_every_file_tree_mutation(tmp_path: Path, mutation: str):
    task = _task(tmp_path)
    manifest = BenchmarkIntegrityGuard.capture(task)
    if mutation == "content":
        (task / "judge.py").write_text("# changed\n")
    elif mutation == "add":
        (task / "new.txt").write_text("new")
    elif mutation == "remove":
        (task / "assets" / "message.txt").unlink()
    else:
        os.chmod(task / "judge.py", 0o700)
    with pytest.raises(IntegrityError):
        BenchmarkIntegrityGuard.verify(task, manifest)


def test_no_follow_copy_matches_full_manifest(tmp_path: Path):
    task = _task(tmp_path)
    expected = BenchmarkIntegrityGuard.capture(task)
    destination = tmp_path / "copy"
    safe_copy_tree(task, destination, expected_manifest=expected)
    assert BenchmarkIntegrityGuard.capture(destination) == expected
    BenchmarkIntegrityGuard.verify(task, expected)


def test_manifest_and_copy_reject_symlink_and_hardlink(tmp_path: Path):
    task = _task(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("secret")
    (task / "link").symlink_to(outside)
    with pytest.raises(IntegrityError):
        BenchmarkIntegrityGuard.capture(task)
    (task / "link").unlink()
    os.link(task / "judge.py", task / "judge-alias.py")
    with pytest.raises(IntegrityError):
        BenchmarkIntegrityGuard.capture(task)
