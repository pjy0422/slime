"""Versioned, resumable records at the DTAP rollout/training boundary."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "dtap-training-record"
VERSION = 1
ELIGIBLE_STATUSES = frozenset({"succeeded", "exhausted", "policy_limit"})
INELIGIBLE_STATUSES = frozenset({"infra_error", "security_abort"})
ARTIFACT_KINDS = frozenset(
    {
        "policy_prompt",
        "policy_response",
        "policy_trajectory",
        "mcp_trajectory",
        "victim_trajectory",
        "judge_result",
    }
)
_SAFE_WORKER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_METADATA_KEYS = frozenset(
    {
        "m4_schema",
        "dtap_status",
        "submissions_used",
        "max_submissions",
        "submit_calls",
        "max_submit_calls",
        "sample_id",
        "task_id",
        "domain",
        "dataset_path",
        "feedback_mode",
        "hae_policy_mode",
        "abort_reason",
        "truncated",
    }
)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().tolist()
    raise TypeError(f"unsupported training record value: {type(value).__name__}")


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def _validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    digest = payload.pop("payload_sha256", None)
    if payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise ValueError("unsupported training record")
    if hashlib.sha256(_canonical(payload)).hexdigest() != digest:
        raise ValueError("training record digest mismatch")
    if not isinstance(payload.get("outcome", {}).get("eligible"), bool):
        raise ValueError("training record eligibility is missing")
    payload["payload_sha256"] = digest
    return payload


def task_reference(metadata: Mapping[str, Any]) -> str:
    for key in ("dataset_path", "task_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            return value
    domain = metadata.get("domain")
    sample_id = metadata.get("sample_id")
    if isinstance(domain, str) and domain and isinstance(sample_id, (str, int)):
        return f"{domain}/{sample_id}"
    task_dir = str(metadata.get("task_dir") or "unidentified-task")
    return f"task-sha256:{hashlib.sha256(task_dir.encode()).hexdigest()}"


def classify_eligibility(status: str, failure_class: str | None = None) -> tuple[bool, str]:
    if failure_class == "unsupported_placement":
        return False, "unsupported_placement"
    if status in ELIGIBLE_STATUSES:
        return True, "attack_outcome"
    if status in INELIGIBLE_STATUSES:
        return False, status
    return False, "incomplete"


@dataclass(frozen=True)
class TrainingRecordContext:
    training_run_id: str
    task_ref: str
    public_episode_id: str
    seed: int | None = None
    tuning_trial_id: str | None = None
    runtime_setup_digest: str | None = None
    feedback_mode: str | None = None
    hierarchy_mode: str | None = None
    worker_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("training_run_id", "task_ref", "public_episode_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                raise ValueError(f"{field_name} must be a non-empty, non-absolute reference")
        if self.worker_id is not None and (
            not _SAFE_WORKER_ID.fullmatch(self.worker_id) or self.worker_id in {".", ".."}
        ):
            raise ValueError("worker_id must be one safe path component")


@dataclass(frozen=True)
class ArtifactReferenceRequest:
    """Trusted lookup key passed to a launcher-owned artifact provider."""

    training_run_id: str
    worker_id: str
    record_id: str
    public_episode_id: str
    private_adapter_session_id: str = field(repr=False)


def validate_artifact_references(raw: Any, *, worker_id: str | None) -> list[dict[str, Any]]:
    """Validate content-free, worker-scoped references supplied by a launcher."""

    if raw is None:
        return []
    if worker_id is None:
        raise ValueError("artifact references require a worker_id")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) > 32:
        raise ValueError("artifact references must be a bounded sequence")
    prefix = ("workers", worker_id)
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    allowed = {"kind", "ref", "sha256", "size_bytes", "media_type", "attempt_index"}
    for item in raw:
        if not isinstance(item, Mapping) or set(item) - allowed:
            raise ValueError("invalid artifact reference fields")
        kind = item.get("kind")
        ref = item.get("ref")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if kind not in ARTIFACT_KINDS:
            raise ValueError("unknown artifact reference kind")
        if not isinstance(ref, str) or not ref or len(ref.encode("utf-8")) > 1024 or "\\" in ref:
            raise ValueError("invalid artifact reference")
        path = PurePosixPath(ref)
        if path.is_absolute() or path.parts[:2] != prefix or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("artifact reference escapes its worker namespace")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("artifact reference sha256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("artifact reference size is invalid")
        key = (str(kind), ref)
        if key in seen:
            raise ValueError("duplicate artifact reference")
        seen.add(key)
        output: dict[str, Any] = {"kind": kind, "ref": ref, "sha256": digest, "size_bytes": size}
        media_type = item.get("media_type")
        if media_type is not None:
            if (
                not isinstance(media_type, str)
                or len(media_type) > 127
                or "/" not in media_type
                or any(char.isspace() for char in media_type)
            ):
                raise ValueError("artifact reference media_type is invalid")
            output["media_type"] = media_type
        attempt_index = item.get("attempt_index")
        if attempt_index is not None:
            if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
                raise ValueError("artifact reference attempt_index is invalid")
            output["attempt_index"] = attempt_index
        normalized.append(output)
    return sorted(normalized, key=lambda item: (item["kind"], item["ref"]))


def sample_payload(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None) or {}
    safe_metadata = {key: _plain(metadata[key]) for key in SAFE_METADATA_KEYS if key in metadata}
    return {
        "group_index": getattr(sample, "group_index", None),
        "index": getattr(sample, "index", None),
        "rollout_id": getattr(sample, "rollout_id", None),
        "tokens": _plain(getattr(sample, "tokens", []) or []),
        "response_length": int(getattr(sample, "response_length", 0) or 0),
        "reward": _plain(getattr(sample, "reward", None)),
        "loss_mask": _plain(getattr(sample, "loss_mask", None)),
        "rollout_log_probs": _plain(getattr(sample, "rollout_log_probs", None)),
        "rollout_top_p_token_ids": _plain(getattr(sample, "rollout_top_p_token_ids", None)),
        "rollout_top_p_token_offsets": _plain(getattr(sample, "rollout_top_p_token_offsets", None)),
        "rollout_routed_experts": _plain(getattr(sample, "rollout_routed_experts", None)),
        "status": _plain(getattr(sample, "status", "pending")),
        "metadata": safe_metadata,
        "train_metadata": _plain(getattr(sample, "train_metadata", None)),
        "remove_sample": bool(getattr(sample, "remove_sample", False)),
    }


def build_record(
    *,
    record_id: str,
    context: TrainingRecordContext,
    status: str,
    failure_class: str | None,
    samples: Sequence[Any],
    reproduction: Mapping[str, Any] | None = None,
    episode: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eligible, reason = classify_eligibility(status, failure_class)
    record = {
        "schema": SCHEMA,
        "version": VERSION,
        "record_id": record_id,
        "context": _plain(context.__dict__),
        "outcome": {
            "status": status,
            "failure_class": failure_class,
            "eligible": eligible,
            "eligibility_reason": reason,
        },
        "reproduction": _plain(dict(reproduction or {})),
        "episode": _plain(dict(episode or {})),
        "samples": [sample_payload(sample) for sample in samples] if eligible else [],
    }
    record["payload_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return record


class TrainingRecordStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @staticmethod
    def record_id(context: TrainingRecordContext, sample: Any) -> str:
        identity = {
            "training_run_id": context.training_run_id,
            "task_ref": context.task_ref,
            "seed": context.seed,
            "group_index": getattr(sample, "group_index", None),
            "index": getattr(sample, "index", None),
            "rollout_id": getattr(sample, "rollout_id", None),
            "sample_id": (getattr(sample, "metadata", None) or {}).get("sample_id"),
        }
        return hashlib.sha256(_canonical(identity)).hexdigest()

    def _paths(self, record_id: str) -> tuple[Path, Path]:
        if len(record_id) != 64 or any(char not in "0123456789abcdef" for char in record_id):
            raise ValueError("record_id must be a lowercase sha256 digest")
        return self.root / "eligible" / f"{record_id}.json", self.root / "excluded" / f"{record_id}.json"

    def load(self, record_id: str) -> dict[str, Any] | None:
        matches = [path for path in self._paths(record_id) if path.exists()]
        if not matches:
            return None
        if len(matches) != 1 or matches[0].is_symlink():
            raise ValueError("ambiguous or unsafe training record")
        return _validate_record(json.loads(matches[0].read_text(encoding="utf-8")))

    def write(self, record: Mapping[str, Any]) -> Path:
        record = _validate_record(record)
        record_id = str(record.get("record_id", ""))
        eligible, excluded = self._paths(record_id)
        target = eligible if record.get("outcome", {}).get("eligible") is True else excluded
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _canonical(record)
        fd, temp_name = tempfile.mkstemp(prefix=f".{record_id}.", dir=target.parent)
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp, target)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                if target.is_symlink() or not target.is_file():
                    raise ValueError("unsafe existing training record") from exc
                if target.read_bytes() != data:
                    raise ValueError("training record identity collision") from exc
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return target
        finally:
            temp.unlink(missing_ok=True)


def restore_samples(record: Mapping[str, Any]) -> list[Any]:
    if record.get("outcome", {}).get("eligible") is not True:
        return []
    from slime.utils.types import Sample

    return [Sample.from_dict(dict(payload)) for payload in record.get("samples", [])]


__all__ = [
    "ARTIFACT_KINDS",
    "ArtifactReferenceRequest",
    "SCHEMA",
    "VERSION",
    "TrainingRecordContext",
    "TrainingRecordStore",
    "build_record",
    "classify_eligibility",
    "restore_samples",
    "sample_payload",
    "task_reference",
    "validate_artifact_references",
]
