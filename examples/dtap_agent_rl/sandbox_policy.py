"""Fail-closed attestation contract for the M4 policy sandbox."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class SandboxPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxSecurityProfile:
    non_root: bool
    no_linux_capabilities: bool
    read_only_root: bool
    isolated_home: bool
    isolated_workdir: bool
    proc_isolated: bool
    no_docker_socket: bool
    no_host_workspace: bool
    network_default_deny: bool
    process_group_cleanup: bool
    allowed_endpoints: frozenset[str]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SandboxSecurityProfile:
        boolean_fields = (
            "non_root",
            "no_linux_capabilities",
            "read_only_root",
            "isolated_home",
            "isolated_workdir",
            "proc_isolated",
            "no_docker_socket",
            "no_host_workspace",
            "network_default_deny",
            "process_group_cleanup",
        )
        values = {}
        for name in boolean_fields:
            if raw.get(name) is not True:
                raise SandboxPolicyError("sandbox security attestation is incomplete")
            values[name] = True
        endpoints = raw.get("allowed_endpoints")
        if not isinstance(endpoints, (list, tuple, set, frozenset)):
            raise SandboxPolicyError("sandbox endpoint attestation is missing")
        return cls(**values, allowed_endpoints=frozenset(str(item) for item in endpoints))


class SandboxPolicyVerifier:
    async def verify(self, sandbox: Any, *, expected_endpoints: frozenset[str]) -> None:
        provider = getattr(sandbox, "attest_m4_security", None)
        if provider is not None:
            raw = provider()
            if inspect.isawaitable(raw):
                raw = await raw
        else:
            raw = getattr(sandbox, "security_profile", None)
        if not isinstance(raw, Mapping):
            raise SandboxPolicyError("sandbox does not provide M4 security attestation")
        profile = SandboxSecurityProfile.from_mapping(raw)
        if profile.allowed_endpoints != expected_endpoints:
            raise SandboxPolicyError("sandbox egress allowlist does not match M4 endpoints")
