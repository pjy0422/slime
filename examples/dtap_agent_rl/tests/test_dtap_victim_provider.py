import os
from unittest.mock import patch

import pytest

from examples.dtap_agent_rl.scripts.run_dtap_attempt import _configure_victim_provider


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
        _configure_victim_provider()
        assert os.environ["ANTHROPIC_BASE_URL"] == "https://provider.example"
        assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "shared-provider-key"
        assert "DTAP_VICTIM_ANTHROPIC_BASE_URL" not in os.environ
        assert "DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN" not in os.environ


def test_victim_provider_rejects_invalid_url_or_missing_key():
    with patch.dict(os.environ, {"DTAP_VICTIM_ANTHROPIC_BASE_URL": "file:///tmp/socket"}, clear=True):
        with pytest.raises(RuntimeError, match="provider URL"):
            _configure_victim_provider()
    with patch.dict(os.environ, {"DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN": "1"}, clear=True):
        with pytest.raises(RuntimeError, match="API key"):
            _configure_victim_provider()
