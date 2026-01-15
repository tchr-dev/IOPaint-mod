"""Configuration for external image services (stubs)."""

from dataclasses import dataclass, field
import os


@dataclass
class ExternalImageServiceConfig:
    """External image service configuration.

    Environment Variables:
        AIE_UPSCALE_SERVICE_URL: Base URL for upscale service
        AIE_UPSCALE_SERVICE_API_KEY: API key for upscale service
        AIE_BG_REMOVE_SERVICE_URL: Base URL for background removal service
        AIE_BG_REMOVE_SERVICE_API_KEY: API key for background removal service
    """

    upscale_url: str = field(
        default_factory=lambda: os.getenv("AIE_UPSCALE_SERVICE_URL", "")
    )
    upscale_api_key: str = field(
        default_factory=lambda: os.getenv("AIE_UPSCALE_SERVICE_API_KEY", "")
    )
    background_remove_url: str = field(
        default_factory=lambda: os.getenv("AIE_BG_REMOVE_SERVICE_URL", "")
    )
    background_remove_api_key: str = field(
        default_factory=lambda: os.getenv("AIE_BG_REMOVE_SERVICE_API_KEY", "")
    )

    @property
    def upscale_enabled(self) -> bool:
        return bool(self.upscale_url and self.upscale_api_key)

    @property
    def background_remove_enabled(self) -> bool:
        return bool(self.background_remove_url and self.background_remove_api_key)
