from iopaint.openai_compat.capabilities import build_openai_capabilities
from iopaint.openai_compat.models import OpenAIModelInfo, ImageSize, ImageQuality


def test_capabilities_normalization():
    models = [
        OpenAIModelInfo(id="openai/gpt-image-1", object="model", created=1, owned_by="openai"),
        OpenAIModelInfo(id="dall-e-2", object="model", created=1, owned_by="openai"),
        OpenAIModelInfo(id="openai/dall-e-3", object="model", created=1, owned_by="openai"),
        OpenAIModelInfo(id="gpt-4o", object="model", created=1, owned_by="openai"),
    ]

    capabilities = build_openai_capabilities(models)

    generate_models = capabilities.modes["images_generate"].models
    edit_models = capabilities.modes["images_edit"].models

    assert [model.id for model in generate_models] == [
        "gpt-image-1",
        "dall-e-2",
        "dall-e-3",
    ]
    assert [model.id for model in edit_models] == [
        "gpt-image-1",
        "dall-e-2",
        "dall-e-3",
    ]

    gpt_model = generate_models[0]
    assert gpt_model.api_id == "openai/gpt-image-1"
    assert gpt_model.default_size == ImageSize.SIZE_1024
    assert gpt_model.default_quality == ImageQuality.STANDARD
    assert ImageSize.SIZE_1792_1024 in gpt_model.sizes
    assert ImageQuality.HD in gpt_model.qualities
