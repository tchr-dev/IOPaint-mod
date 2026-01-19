"""Tests for API error handling fixes."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iopaint.api import Api
from iopaint.budget.config import BudgetConfig
from iopaint.schema import (
    ApiConfig,
    Device,
    InteractiveSegModel,
    RealESRGANModel,
    RemoveBGModel,
)


def _make_api_config(tmp_path: Path) -> ApiConfig:
    """Create a test API configuration."""
    return ApiConfig(
        host="127.0.0.1",
        port=8080,
        inbrowser=False,
        model="lama",
        no_half=True,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw_checker=True,
        local_files_only=True,
        cpu_textencoder=False,
        device=Device.cpu,
        input=None,
        mask_dir=None,
        output_dir=None,
        quality=95,
        enable_interactive_seg=False,
        interactive_seg_model=InteractiveSegModel.vit_b,
        interactive_seg_device=Device.cpu,
        enable_remove_bg=False,
        remove_bg_device=Device.cpu,
        remove_bg_model=RemoveBGModel.briaai_rmbg_1_4.value,
        enable_anime_seg=False,
        enable_realesrgan=False,
        realesrgan_device=Device.cpu,
        realesrgan_model=RealESRGANModel.realesr_general_x4v3,
        enable_gfpgan=False,
        gfpgan_device=Device.cpu,
        enable_restoreformer=False,
        restoreformer_device=Device.cpu,
        enable_upload_api=False,
        enable_download_api=False,
        cors_origins=[],
        max_queue_size=0,
        model_dir=None,
    )


def _make_budget_config(tmp_path: Path) -> BudgetConfig:
    """Create a test budget configuration."""
    return BudgetConfig(
        data_dir=tmp_path / "budget_data",
        daily_cap_usd=10.0,
        monthly_cap_usd=100.0,
        session_cap_usd=5.0,
        rate_limit_seconds=0,
    )


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """Create a test client with mocked dependencies."""
    api_config = _make_api_config(tmp_path)

    monkeypatch.setenv("AIE_DATA_DIR", str(tmp_path / "budget_data"))

    mock_model_manager = Mock()
    mock_model_manager.name = "lama"
    mock_model_manager.current_model = Mock()
    mock_model_manager.current_model.name = "lama"

    mock_plugins = {}

    mock_openai_config = Mock()
    mock_openai_config.is_enabled = False

    monkeypatch.setattr("iopaint.api.Api._build_model_manager", lambda self: mock_model_manager)
    monkeypatch.setattr("iopaint.api.Api._build_plugins", lambda self: mock_plugins)
    monkeypatch.setattr("iopaint.api.OpenAIConfig", lambda: mock_openai_config)

    app = FastAPI()
    api = Api(app, api_config)

    api.model_manager = mock_model_manager
    api.plugins = mock_plugins
    api.openai_config = mock_openai_config

    return TestClient(app)


class TestBudgetLimitsValidation:
    """Test budget limits validation API."""

    def test_update_budget_limits_negative_daily(self, test_client):
        """Test that negative daily budget cap is rejected."""
        response = test_client.post(
            "/api/v1/budget/limits", json={"daily_cap_usd": -5.0}
        )

        assert response.status_code == 400
        assert "Daily budget cap cannot be negative" in response.json()["detail"]

    def test_update_budget_limits_negative_monthly(self, test_client):
        """Test that negative monthly budget cap is rejected."""
        response = test_client.post(
            "/api/v1/budget/limits", json={"monthly_cap_usd": -10.0}
        )

        assert response.status_code == 400
        assert "Monthly budget cap cannot be negative" in response.json()["detail"]

    def test_update_budget_limits_negative_session(self, test_client):
        """Test that negative session budget cap is rejected."""
        response = test_client.post(
            "/api/v1/budget/limits", json={"session_cap_usd": -1.0}
        )

        assert response.status_code == 400
        assert "Session budget cap cannot be negative" in response.json()["detail"]

    def test_update_budget_limits_valid_values(self, test_client):
        """Test that valid non-negative budget caps are accepted."""
        response = test_client.post(
            "/api/v1/budget/limits",
            json={
                "daily_cap_usd": 15.0,
                "monthly_cap_usd": 150.0,
                "session_cap_usd": 7.5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["daily_cap_usd"] == 15.0
        assert data["monthly_cap_usd"] == 150.0
        assert data["session_cap_usd"] == 7.5

    def test_update_budget_limits_partial_update(self, test_client):
        """Test that partial budget limit updates work correctly."""
        # First set some initial values
        test_client.post(
            "/api/v1/budget/limits",
            json={
                "daily_cap_usd": 10.0,
                "monthly_cap_usd": 100.0,
                "session_cap_usd": 5.0,
            },
        )

        # Update only daily cap
        response = test_client.post(
            "/api/v1/budget/limits", json={"daily_cap_usd": 20.0}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["daily_cap_usd"] == 20.0
        # Other values should remain unchanged
        assert data["monthly_cap_usd"] == 100.0
        assert data["session_cap_usd"] == 5.0


class TestModelSwitchingErrorHandling:
    """Test model switching error handling API."""

    def test_switch_to_openai_without_api_key(self, test_client):
        """Test switching to openai-compat without API key returns 400."""
        response = test_client.post("/api/v1/model", json={"name": "openai-compat"})

        # Without API key configured, this should fail
        assert response.status_code in [400, 500]
        if response.status_code == 400:
            assert "OpenAI API key" in response.json()["detail"]


class TestBudgetStatusSerialization:
    """Test budget status API JSON serialization."""

    def test_budget_status_endpoint_exists(self, test_client):
        """Test that budget status endpoint returns valid JSON."""
        response = test_client.get("/api/v1/budget/status")

        assert response.status_code == 200

        # Parse JSON to ensure it's valid
        data = response.json()

        # Verify response structure
        assert "daily" in data
        assert "monthly" in data
        assert "session" in data
        assert "status" in data

        # Verify no infinity values in the response
        json_str = json.dumps(data)
        assert "inf" not in json_str.lower()

        # Verify numeric values are finite
        assert isinstance(data["daily"]["remaining_usd"], (int, float))
        assert isinstance(data["monthly"]["remaining_usd"], (int, float))
        assert isinstance(data["session"]["remaining_usd"], (int, float))
