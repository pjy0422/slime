"""M1 orchestration: sanitized read-only episode -> Claude Code harness."""

from __future__ import annotations

from .attack_surface import ToolCatalogProvider
from .episode import TaskSnapshot
from .harness import DTAPClaudeCodeHarness
from .runtime import registered_episode
from .service import EpisodeRegistry, build_episode_view
from .task_projection import ProjectionPolicy


async def run_m1_episode(
    *,
    snapshot: TaskSnapshot,
    catalog_provider: ToolCatalogProvider,
    registry: EpisodeRegistry,
    sandbox,
    session_id: str,
    adapter_url: str,
    prompt: str,
    workdir: str,
    time_budget_sec: int,
    projection_policy: ProjectionPolicy = ProjectionPolicy(),
    harness=None,
) -> int:
    """Run one read-only DTAP RL episode without exposing privileged task state.

    The episode capability is live only while Claude Code runs. The caller is
    responsible for keeping the shared FastMCP server reachable at
    ``DTAP_HARNESS_URL``.
    """

    view = await build_episode_view(
        snapshot,
        catalog_provider,
        projection_policy=projection_policy,
    )
    harness = harness or DTAPClaudeCodeHarness()

    with registered_episode(registry, token=session_id, view=view):
        return await harness.run(
            sandbox,
            workdir=workdir,
            session_id=session_id,
            adapter_url=adapter_url,
            time_budget_sec=time_budget_sec,
            prompt=prompt,
        )
