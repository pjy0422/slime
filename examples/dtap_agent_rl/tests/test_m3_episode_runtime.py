import pytest

from examples.dtap_agent_rl.episode_runtime import (
    EpisodeRuntimeState,
    EpisodeStatus,
    EpisodeTerminalError,
)


def test_new_episode_is_active_with_full_h_budget():
    state = EpisodeRuntimeState(max_submissions=3)

    assert state.status is EpisodeStatus.ACTIVE
    assert state.submissions_used == 0
    assert state.remaining_submissions == 3
    assert state.successful_attempt is None
    assert state.terminal is False


def test_only_started_dtap_evaluation_consumes_h():
    state = EpisodeRuntimeState(max_submissions=2)

    # Validation and YAML gates run before this transition and therefore do not
    # call mark_evaluation_started().
    assert state.submissions_used == 0
    attempt = state.mark_evaluation_started()

    assert attempt == 1
    assert state.submissions_used == 1
    assert state.remaining_submissions == 1
    assert state.victim_runs_started == 1
    assert state.remaining_victim_runs == 1


def test_success_terminates_immediately_without_spending_remaining_budget():
    state = EpisodeRuntimeState(max_submissions=3)
    attempt = state.mark_evaluation_started()

    state.record_attack_result(attempt_index=attempt, attack_success=True)

    assert state.status is EpisodeStatus.SUCCEEDED
    assert state.terminal is True
    assert state.successful_attempt == 1
    assert state.remaining_submissions == 2
    with pytest.raises(EpisodeTerminalError):
        state.mark_evaluation_started()


def test_hth_normal_failure_exhausts_episode():
    state = EpisodeRuntimeState(max_submissions=2)

    first = state.mark_evaluation_started()
    state.record_attack_result(attempt_index=first, attack_success=False)
    assert state.status is EpisodeStatus.ACTIVE

    second = state.mark_evaluation_started()
    state.record_attack_result(attempt_index=second, attack_success=False)

    assert state.status is EpisodeStatus.EXHAUSTED
    assert state.terminal is True
    assert state.remaining_submissions == 0
    with pytest.raises(EpisodeTerminalError):
        state.mark_evaluation_started()


def test_infrastructure_failure_is_terminal_and_excluded_from_training():
    state = EpisodeRuntimeState(max_submissions=3)

    state.record_infrastructure_failure(stage="docker_start")

    assert state.status is EpisodeStatus.INFRA_ERROR
    assert state.terminal is True
    assert state.remove_sample is True
    assert state.final_reward is None


@pytest.mark.parametrize("max_submissions", [0, -1, True])
def test_h_must_be_a_positive_integer(max_submissions):
    with pytest.raises(ValueError):
        EpisodeRuntimeState(max_submissions=max_submissions)
