import time
import time
from typing import Dict, List, TypedDict, cast

from .models import (
    ImageQuality,
    ImageSize,
    OpenAICapabilitiesResponse,
    OpenAICapabilityModel,
    OpenAIModeCapabilities,
    OpenAIModelInfo,
)


class ModelCapabilitySpec(TypedDict):
    sizes: List[ImageSize]
    qualities: List[ImageQuality]
    default_size: ImageSize
    default_quality: ImageQuality
    modes: List[str]


MODEL_CAPABILITIES: Dict[str, ModelCapabilitySpec] = {
    "gpt-image-1": {
        "sizes": [
            ImageSize.SIZE_1024,
            ImageSize.SIZE_1792_1024,
            ImageSize.SIZE_1024_1792,
        ],
        "qualities": [ImageQuality.STANDARD, ImageQuality.HD],
        "default_size": ImageSize.SIZE_1024,
        "default_quality": ImageQuality.STANDARD,
        "modes": ["images_generate", "images_edit"],
    },
    "dall-e-2": {
        "sizes": [
            ImageSize.SIZE_256,
            ImageSize.SIZE_512,
            ImageSize.SIZE_1024,
        ],
        "qualities": [ImageQuality.STANDARD],
        "default_size": ImageSize.SIZE_1024,
        "default_quality": ImageQuality.STANDARD,
        "modes": ["images_generate", "images_edit"],
    },
    "dall-e-3": {
        "sizes": [
            ImageSize.SIZE_1024,
            ImageSize.SIZE_1792_1024,
            ImageSize.SIZE_1024_1792,
        ],
        "qualities": [ImageQuality.STANDARD, ImageQuality.HD],
        "default_size": ImageSize.SIZE_1024,
        "default_quality": ImageQuality.STANDARD,
        "modes": ["images_generate", "images_edit"],
    },
}


def normalize_model_id(model_id: str) -> str:
    """Normalize model id by removing provider prefixes."""
    return model_id.split("/")[-1]


def _select_api_id(raw_ids: List[str], canonical_id: str) -> str:
    if canonical_id in raw_ids:
        return canonical_id
    for raw_id in raw_ids:
        if raw_id.endswith(f"/{canonical_id}"):
            return raw_id
    return raw_ids[0]


def build_openai_capabilities(
    models: List[OpenAIModelInfo],
) -> OpenAICapabilitiesResponse:
    """Build capabilities response from available model list."""
    available: Dict[str, List[str]] = {}
    for model in models:
        canonical = normalize_model_id(model.id)
        if canonical not in MODEL_CAPABILITIES:
            continue
        available.setdefault(canonical, []).append(model.id)

    modes: Dict[str, OpenAIModeCapabilities] = {
        "images_generate": OpenAIModeCapabilities(),
        "images_edit": OpenAIModeCapabilities(),
    }

    for canonical_id, spec in MODEL_CAPABILITIES.items():
        raw_ids = available.get(canonical_id)
        if not raw_ids:
            continue
        api_id = _select_api_id(raw_ids, canonical_id)
        sizes = cast(List[ImageSize], spec["sizes"])
        qualities = cast(List[ImageQuality], spec["qualities"])
        default_size = cast(ImageSize, spec["default_size"])
        default_quality = cast(ImageQuality, spec["default_quality"])
        capability_model = OpenAICapabilityModel(
            id=canonical_id,
            api_id=api_id,
            label=canonical_id,
            sizes=sizes,
            qualities=qualities,
            default_size=default_size,
            default_quality=default_quality,
        )
        for mode in spec["modes"]:
            modes[mode].models.append(capability_model)

    for mode in modes.values():
        if mode.models:
            mode.default_model = mode.models[0].id

    return OpenAICapabilitiesResponse(created=int(time.time()), modes=modes)
