import json
from unittest.mock import patch

import pytest

from examples.dtap_agent_rl.feedback.providers import AnthropicMessagesJSONCompleter


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps({
            "content": [{"type": "text", "text": "```json\n{\"answer\": true}\n```"}],
            "usage": {"input_tokens": 11, "output_tokens": 4},
        }).encode()


@pytest.mark.asyncio
async def test_anthropic_completer_keeps_key_in_headers_and_records_bounded_metrics():
    completer = AnthropicMessagesJSONCompleter(
        base_url="https://provider.example/v1", api_key="private-key",
        model="glm-5.2", timeout_seconds=3,
    )
    with patch("urllib.request.urlopen", return_value=_Response()) as opened:
        assert await completer("untrusted trace") == {"answer": True}

    request = opened.call_args.args[0]
    assert request.full_url == "https://provider.example/v1/messages"
    assert request.headers["X-api-key"] == "private-key"
    assert b"private-key" not in request.data
    assert completer.usage.to_dict() | {"latency_seconds": 0.0} == {
        "calls": 1, "failures": 0, "input_tokens": 11, "output_tokens": 4,
        "latency_seconds": 0.0, "cost_usd": None,
    }


@pytest.mark.asyncio
async def test_anthropic_completer_counts_failure_without_leaking_key():
    completer = AnthropicMessagesJSONCompleter(
        base_url="http://127.0.0.1:9999", api_key="private-key", model="glm-5.2",
    )
    with patch("urllib.request.urlopen", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            await completer("trace")
    assert completer.usage.calls == 1
    assert completer.usage.failures == 1


def test_anthropic_completer_rejects_unencrypted_remote_endpoint():
    with pytest.raises(ValueError, match="HTTPS"):
        AnthropicMessagesJSONCompleter(
            base_url="http://provider.example", api_key="key", model="glm-5.2",
        )
