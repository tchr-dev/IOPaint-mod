from iopaint.services.config import ExternalImageServiceConfig


def test_external_services_config_defaults(monkeypatch):
    monkeypatch.delenv("AIE_UPSCALE_SERVICE_URL", raising=False)
    monkeypatch.delenv("AIE_UPSCALE_SERVICE_API_KEY", raising=False)
    monkeypatch.delenv("AIE_BG_REMOVE_SERVICE_URL", raising=False)
    monkeypatch.delenv("AIE_BG_REMOVE_SERVICE_API_KEY", raising=False)

    config = ExternalImageServiceConfig()

    assert config.upscale_enabled is False
    assert config.background_remove_enabled is False


def test_external_services_config_enabled(monkeypatch):
    monkeypatch.setenv("AIE_UPSCALE_SERVICE_URL", "https://example.com/upscale")
    monkeypatch.setenv("AIE_UPSCALE_SERVICE_API_KEY", "key")
    monkeypatch.setenv("AIE_BG_REMOVE_SERVICE_URL", "https://example.com/remove")
    monkeypatch.setenv("AIE_BG_REMOVE_SERVICE_API_KEY", "key")

    config = ExternalImageServiceConfig()

    assert config.upscale_enabled is True
    assert config.background_remove_enabled is True
