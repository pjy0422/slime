import pytest

from examples.dtap_agent_rl.trajectory import run_h_turn_episode


class FakeAdapter:
    def __init__(self):
        self.opened = []
        self.finished = []

    async def open_session(self, **kwargs):
        self.opened.append(kwargs)
        return "same-policy-session"

    async def finish_session(self, session_id, *, reward=None, remove_sample=False):
        self.finished.append((session_id, reward, remove_sample))
        return ["sample"]


class FakePolicy:
    def __init__(self, observations):
        self.observations = list(observations)
        self.session_ids = []

    async def run_until_terminal(self, *, session_id, submit):
        self.session_ids.append(session_id)
        receipts = []
        for observation in self.observations:
            receipts.append(await submit(observation))
            if receipts[-1]["terminal"]:
                break
        return receipts


class FakeController:
    def __init__(self, receipts, *, reward, remove_sample=False):
        self.receipts = iter(receipts)

        class Runtime:
            def __init__(self):
                self.final_reward = reward
                self.remove_sample = remove_sample
                self.terminal = reward is not None or remove_sample
                self.status = type("Status", (), {"value": "test"})()
                self.submissions_used = 0

            def record_infrastructure_failure(self, *, stage):
                self.final_reward = None
                self.remove_sample = True
                self.terminal = True
                self.status = type("Status", (), {"value": "infra_error"})()

        self.runtime = Runtime()

    async def submit(self, _plan):
        return next(self.receipts)


@pytest.mark.asyncio
async def test_h_macro_submissions_share_one_policy_session_and_finish_once():
    adapter = FakeAdapter()
    policy = FakePolicy([{"steps": [1]}, {"steps": [2]}, {"steps": [3]}])
    controller = FakeController(
        [
            {"accepted": True, "success": False, "terminal": False},
            {"accepted": True, "success": False, "terminal": False},
            {"accepted": True, "success": True, "terminal": True},
        ],
        reward=1.0,
    )

    await run_h_turn_episode(adapter=adapter, policy=policy, controller=controller)

    assert len(adapter.opened) == 1
    assert policy.session_ids == ["same-policy-session"]
    assert adapter.finished == [("same-policy-session", 1.0, False)]


@pytest.mark.asyncio
async def test_infrastructure_abort_finishes_once_with_remove_sample_not_zero_reward():
    adapter = FakeAdapter()
    policy = FakePolicy([{"steps": [1]}])
    controller = FakeController(
        [{"accepted": False, "terminal": True, "errors": [{"code": "INFRA_ERROR"}]}],
        reward=None,
        remove_sample=True,
    )

    await run_h_turn_episode(adapter=adapter, policy=policy, controller=controller)

    assert adapter.finished == [("same-policy-session", None, True)]


@pytest.mark.asyncio
async def test_policy_exception_still_finishes_once_and_removes_sample():
    adapter = FakeAdapter()

    class BrokenPolicy:
        async def run_until_terminal(self, *, session_id, submit):
            raise RuntimeError("policy crashed")

    controller = FakeController([], reward=None)

    with pytest.raises(RuntimeError, match="policy crashed"):
        await run_h_turn_episode(adapter=adapter, policy=BrokenPolicy(), controller=controller)

    assert adapter.finished == [("same-policy-session", None, True)]


@pytest.mark.asyncio
async def test_current_slime_adapter_signature_opens_and_finishes_once():
    class Sample:
        remove_sample = False

    class SlimeAdapter:
        def __init__(self):
            self.opened = []
            self.finished = []
            self.dropped = []

        def open_session(self, session_id, **kwargs):
            self.opened.append((session_id, kwargs))

        async def finish_session(self, session_id, **kwargs):
            self.finished.append((session_id, kwargs))
            return [Sample()]

        async def drop_session(self, session_id, *, wait_timeout):
            self.dropped.append((session_id, wait_timeout))

    adapter = SlimeAdapter()
    policy = FakePolicy([{"steps": [1]}])
    controller = FakeController(
        [{"accepted": True, "success": True, "terminal": True}], reward=1.0
    )
    base_sample = Sample()

    samples = await run_h_turn_episode(
        adapter=adapter,
        policy=policy,
        controller=controller,
        session_id="slime-session",
        base_sample=base_sample,
        sampling_defaults={"temperature": 1.0},
        max_context_tokens=4096,
    )

    assert adapter.opened == [(
        "slime-session",
        {"sampling_defaults": {"temperature": 1.0}, "max_context_tokens": 4096},
    )]
    assert adapter.finished[0][0] == "slime-session"
    assert adapter.finished[0][1]["base_sample"] is base_sample
    assert adapter.finished[0][1]["reward"] == 1.0
    assert adapter.dropped == [("slime-session", 30)]
    assert samples[0].remove_sample is False
