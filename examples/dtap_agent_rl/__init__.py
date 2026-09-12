"""DTAP RL integration example for slime (M0 through M4)."""

from .attack_surface import AttackSurface, ToolSpec, build_attack_surface
from .attempt_runner import AttemptResult, DtapAttemptRunner
from .authority import EpisodeAuthorityRegistry, EpisodeCredentials
from .episode import TaskSnapshot, load_task_snapshot
from .episode_runtime import EpisodeRuntimeState, EpisodeStatus
from .integrity import BenchmarkIntegrityGuard, BenchmarkManifest
from .placement import DtapPlacementRunner, PlacementCoordinator, PlacementRunResult
from .scheduler import AttemptScheduler
from .security_policy import M4SecurityPolicy
from .service import EpisodeRegistry, EpisodeView, build_episode_view
from .submission import EpisodeSubmissionRegistry, SubmissionCoordinator, SubmissionPlan
from .task_projection import PolicyTaskSpec, ProjectionPolicy, project_task

__all__ = [
    "AttackSurface",
    "AttemptResult",
    "AttemptScheduler",
    "BenchmarkIntegrityGuard",
    "BenchmarkManifest",
    "DtapAttemptRunner",
    "EpisodeAuthorityRegistry",
    "EpisodeCredentials",
    "EpisodeRegistry",
    "EpisodeRuntimeState",
    "EpisodeStatus",
    "EpisodeSubmissionRegistry",
    "EpisodeView",
    "M4SecurityPolicy",
    "DtapPlacementRunner",
    "PlacementCoordinator",
    "PlacementRunResult",
    "PolicyTaskSpec",
    "ProjectionPolicy",
    "TaskSnapshot",
    "ToolSpec",
    "SubmissionCoordinator",
    "SubmissionPlan",
    "build_attack_surface",
    "build_episode_view",
    "load_task_snapshot",
    "project_task",
]
