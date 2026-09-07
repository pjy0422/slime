from pathlib import Path

import pytest

from examples.dtap_agent_rl.scripts.prepare_macos_baseline import (
    normalize_data_root,
    snapshot_names,
)


def test_normalize_macos_data_root_accepts_root_or_version(tmp_path: Path):
    version = tmp_path / "14"
    version.mkdir()
    (version / "data.qcow2").touch()
    assert normalize_data_root(tmp_path) == tmp_path.resolve()
    assert normalize_data_root(version) == tmp_path.resolve()


def test_normalize_macos_data_root_rejects_missing_image(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="expected data.qcow2"):
        normalize_data_root(tmp_path)


def test_snapshot_names_parses_qemu_table(monkeypatch, tmp_path: Path):
    class Result:
        stdout = (
            "Snapshot list:\n"
            "ID TAG VM_SIZE DATE VM_CLOCK ICOUNT\n"
            "1 booted 3.73 GiB 2026-03-16 00:37:57 00:12:35 --\n"
        )

    monkeypatch.setattr(
        "examples.dtap_agent_rl.scripts.prepare_macos_baseline._qemu_img",
        lambda *args, **kwargs: Result(),
    )
    assert snapshot_names(tmp_path, "image") == {"booted"}
