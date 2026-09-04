import pytest

from examples.dtap_agent_rl.transport_security import M4RequestLimitMiddleware


@pytest.mark.asyncio
async def test_transport_rejects_oversized_stream_before_downstream():
    called = False

    async def downstream(scope, receive, send):
        nonlocal called
        called = True

    messages = iter(
        [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"56789", "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    app = M4RequestLimitMiddleware(downstream, max_body_bytes=8, max_header_bytes=16)
    await app({"type": "http", "headers": []}, receive, send)

    assert called is False
    assert sent[0]["status"] == 413
    assert b"request unavailable" in sent[1]["body"]


@pytest.mark.asyncio
async def test_transport_replays_bounded_body_exactly_once():
    observed = []

    async def downstream(scope, receive, send):
        observed.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    incoming = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return next(incoming)

    async def send(message):
        sent.append(message)

    app = M4RequestLimitMiddleware(downstream, max_body_bytes=8, max_header_bytes=16)
    await app({"type": "http", "headers": []}, receive, send)
    assert observed == [{"type": "http.request", "body": b"abcdef", "more_body": False}]
    assert sent[0]["status"] == 200
