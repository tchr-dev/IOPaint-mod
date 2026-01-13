"""Configuration for OpenAI-compatible API client."""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI-compatible API client.

    Configuration is loaded from environment variables with the following precedence:
    1. Constructor arguments (highest)
    2. Environment variables
    3. Default values (lowest)

    Environment Variables:
        AIE_BACKEND: Backend identifier (default: "openai")
        AIE_OPENAI_API_KEY: API key (required for API calls)
        AIE_OPENAI_BASE_URL: Base URL for API (default: https://api.openai.com/v1)
        AIE_OPENAI_MODEL: Default model for image operations (default: gpt-image-1)
        AIE_OPENAI_TIMEOUT_S: Request timeout in seconds (default: 120)
        AIE_OPENAI_REFINE_MODEL: Model for prompt refinement (default: gpt-4o-mini)
    """

    backend: str = field(
        default_factory=lambda: os.getenv("AIE_BACKEND", "openai")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("AIE_OPENAI_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "AIE_OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv("AIE_OPENAI_MODEL", "gpt-image-1")
    )
    timeout_s: int = field(
        default_factory=lambda: int(os.getenv("AIE_OPENAI_TIMEOUT_S", "120"))
    )
    refine_model: str = field(
        default_factory=lambda: os.getenv("AIE_OPENAI_REFINE_MODEL", "gpt-4o-mini")
    )

    @property
    def is_enabled(self) -> bool:
        """Check if OpenAI client is configured with an API key."""
        return bool(self.api_key)

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required configuration is missing.
        """
        if not self.api_key:
            raise ValueError(
                "AIE_OPENAI_API_KEY environment variable is required. "
                "Set it via environment variable or --openai-api-key CLI flag."
            )
        if not self.base_url:
            raise ValueError("AIE_OPENAI_BASE_URL cannot be empty.")

    def __repr__(self) -> str:
        """Return string representation with masked API key."""
        masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return (
            f"OpenAIConfig("
            f"backend={self.backend!r}, "
            f"api_key={masked_key!r}, "
            f"base_url={self.base_url!r}, "
            f"model={self.model!r}, "
            f"timeout_s={self.timeout_s})"
        )
