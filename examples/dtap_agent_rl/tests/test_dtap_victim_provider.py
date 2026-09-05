import os
from unittest.mock import patch

import pytest

from examples.dtap_agent_rl.scripts.run_dtap_attempt import (
    _configure_victim_provider,
    _restored_environment,
)


def test_victim_provider_aliases_are_consumed_inside_dtap_child():
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "shared-provider-key",
            "DTAP_VICTIM_ANTHROPIC_BASE_URL": "https://provider.example",
            "DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN": "1",
        },
        clear=True,
    ):
        _configure_victim_provider("deepseek-v4-flash")
        assert os.environ["ANTHROPIC_BASE_URL"] == "https://provider.example"
        assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "shared-provider-key"
        assert os.environ["OPENAI_BASE_URL"] == "https://provider.example/v1"
        assert os.environ["OPENAI_API_KEY"] == "shared-provider-key"
        assert os.environ["PATIENT_MODEL"] == "deepseek-v4-flash"
        assert os.environ["MEASUREMENT_MODEL"] == "deepseek-v4-flash"
        assert os.environ["JUDGE_MODEL"] == "deepseek-v4-flash"
        assert os.environ["OPENCLAW_MCP_TOOL_TIMEOUT_SECONDS"] == "600"
        assert "DTAP_VICTIM_ANTHROPIC_BASE_URL" not in os.environ
        assert "DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN" not in os.environ


def test_victim_provider_rejects_invalid_url_or_missing_key():
    with patch.dict(os.environ, {"DTAP_VICTIM_ANTHROPIC_BASE_URL": "file:///tmp/socket"}, clear=True):
        with pytest.raises(RuntimeError, match="provider URL"):
            _configure_victim_provider("deepseek-v4-flash")


def test_victim_credentials_can_be_hidden_during_aux_startup_then_restored():
    with patch.dict(os.environ, {}, clear=True):
        credentials = {"ANTHROPIC_API_KEY": "victim-key"}
        assert "ANTHROPIC_API_KEY" not in os.environ
        with _restored_environment(credentials):
            assert os.environ["ANTHROPIC_API_KEY"] == "victim-key"
        assert "ANTHROPIC_API_KEY" not in os.environ
    with patch.dict(os.environ, {"DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN": "1"}, clear=True):
        with pytest.raises(RuntimeError, match="API key"):
            _configure_victim_provider("deepseek-v4-flash")
