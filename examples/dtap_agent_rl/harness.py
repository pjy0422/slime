"""Minimal DTAP specialization of slime's ClaudeCodeHarness for M1."""

from __future__ import annotations

import json
import os
import shlex

from slime.agent.harness import ClaudeCodeHarness, HarnessContext
from slime.agent.harness.common import run_agent
from slime.agent.sandbox import Sandbox


class DTAPClaudeCodeHarness(ClaudeCodeHarness):
    """Claude Code runtime restricted to the host-side DTAP MCP server.

    This class deliberately does not parse config.yaml and does not know a DTAP
    task directory. Its only DTAP responsibilities are MCP transport config and
    per-episode capability injection.
    """

    name = "dtap_claude_code"

    harness_url_env = "DTAP_HARNESS_URL"
    mcp_config_path = "/home/agent/.dtap/mcp.json"

    # M1 is read-only. Keep the base Claude streaming flags, allow only the DTAP
    # MCP server without prompts, and explicitly deny common native tools. The
    # sandbox must still be treated as untrusted; M4 will harden network/runtime.
    denied_native_tools = (
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
        "NotebookEdit",
    )
    launch_flags = (
        ClaudeCodeHarness.launch_flags
        + f" --mcp-config {mcp_config_path}"
        + ' --allowedTools "mcp__dtap"'
        + f' --disallowedTools "{",".join(denied_native_tools)}"'
    )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        await super().write_config(sb, ctx)

        # Claude Code expands environment variables in HTTP MCP url/headers.
        # Therefore the config file contains no episode token or deployment URL.
        config = {
            "mcpServers": {
                "dtap": {
                    "type": "http",
                    "url": "${DTAP_HARNESS_URL}",
                    "headers": {
                        "Authorization": "Bearer ${DTAP_EPISODE_TOKEN}",
                    },
                }
            }
        }
        await sb.exec(
            "mkdir -p /home/agent/.dtap && "
            f"echo {shlex.quote(json.dumps(config, sort_keys=True))} > {self.mcp_config_path} && "
            "chown -R agent:agent /home/agent/.dtap",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(
        self,
        sb: Sandbox,
        ctx: HarnessContext,
        prompt: str,
        time_budget_sec: int,
    ) -> int:
        harness_url = os.environ.get(self.harness_url_env, "").strip()
        if not harness_url:
            raise RuntimeError(f"{self.harness_url_env} must point to the host-side MCP endpoint")

        cmd = f"/usr/local/bin/claude -p {shlex.quote(prompt)} {self.launch_flags}"
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"

        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_MODEL": ctx.model_label,
            "DTAP_HARNESS_URL": harness_url,
            # M1 reuses slime's opaque session id as an episode capability. It is
            # not written to disk and is absent from MCP tool arguments/schemas.
            "DTAP_EPISODE_TOKEN": ctx.session_id,
            **self.static_env,
        }
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))

        return await run_agent(
            sb,
            workdir=ctx.workdir,
            start_cmd=cmd,
            env=env,
            time_budget_sec=time_budget_sec,
        )
