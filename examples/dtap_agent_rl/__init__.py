"""DTAP RL integration example for slime (M0 + M1)."""

from .attack_surface import AttackSurface, ToolSpec, build_attack_surface
from .episode import TaskSnapshot, load_task_snapshot
from .service import EpisodeRegistry, EpisodeView, build_episode_view
from .task_projection import PolicyTaskSpec, ProjectionPolicy, project_task

__all__ = [
    "AttackSurface",
    "EpisodeRegistry",
    "EpisodeView",
    "PolicyTaskSpec",
    "ProjectionPolicy",
    "TaskSnapshot",
    "ToolSpec",
    "build_attack_surface",
    "build_episode_view",
    "load_task_snapshot",
    "project_task",
]
