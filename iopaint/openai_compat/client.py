"""HTTP client for OpenAI-compatible image APIs."""

import base64
from typing import List, Optional
import time

import httpx
from loguru import logger

from .config import OpenAIConfig
from .errors import (
    OpenAIError,
    classify_error,
    create_timeout_error,
    create_network_error,
)
from .models import (
    OpenAIModelInfo,
    ListModelsResponse,
    EditImageRequest,
    EditImageResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    RefinePromptRequest,
    RefinePromptResponse,
    CreateVariationRequest,
    CreateVariationResponse,
    ImageData,
)


class OpenAICompatClient:
    """HTTP client for OpenAI-compatible image APIs.

    Supports OpenAI, ProxyAPI, OpenRouter, and other compatible providers.

    Example:
        >>> config = OpenAIConfig(api_key="sk-xxx")
        >>> client = OpenAICompatClient(config)
        >>> models = client.list_models()
        >>> result = client.generate_image(GenerateImageRequest(prompt="A cat"))
    """

    def __init__(self, config: OpenAIConfig):
        """Initialize the client.

        Args:
            config: OpenAI configuration with API key and base URL.
        """
        self.config = config
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialized HTTP client."""
        if self._client is None:
            # Normalize base URL (remove trailing slash)
            base_url = self.config.base_url.rstrip("/")

            self._client = httpx.Client(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                },
                timeout=httpx.Timeout(self.config.timeout_s),
            )
        return self._client

    def _check_response(self, response: httpx.Response) -> None:
        """Check response for errors and raise OpenAIError if needed."""
        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": {"message": response.text}}

            raise classify_error(response.status_code, error_body)

    def _make_request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """Make HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/models")
            **kwargs: Additional arguments for httpx request

        Returns:
            httpx.Response object

        Raises:
            OpenAIError: On API or network errors
        """
        try:
            response = self.client.request(method, path, **kwargs)
            self._check_response(response)
            return response
        except OpenAIError:
            raise
        except httpx.TimeoutException as e:
            raise create_timeout_error(e)
        except httpx.RequestError as e:
            raise create_network_error(e)

    # =========================================================================
    # Models
    # =========================================================================

    def list_models(self) -> List[OpenAIModelInfo]:
        """List available models from the API.

        Returns:
            List of available models.

        Raises:
            OpenAIError: On API errors.
        """
        logger.debug("Fetching models list from API")
        response = self._make_request("GET", "/models")
        data = response.json()

        # Filter to image-related models if possible
        models = []
        for model_data in data.get("data", []):
            try:
                models.append(OpenAIModelInfo(**model_data))
            except Exception as e:
                logger.warning(f"Failed to parse model data: {e}")
                continue

        logger.info(f"Found {len(models)} models")
        return models

    # =========================================================================
    # Prompt Refinement
    # =========================================================================

    def refine_prompt(self, request: RefinePromptRequest) -> RefinePromptResponse:
        """Refine/expand a prompt using a cheap LLM call.

        Uses chat completions endpoint to improve image generation prompts
        before sending to expensive image APIs.

        Args:
            request: Refinement request with original prompt.

        Returns:
            Response with refined prompt.

        Raises:
            OpenAIError: On API errors.
        """
        model = request.model or self.config.refine_model

        system_prompt = (
            "You are a helpful assistant that improves image generation prompts. "
            "Expand and enhance the given prompt to be more descriptive and detailed, "
            "but keep it concise (under 200 words). Focus on visual details, style, "
            "lighting, composition, and mood. Output only the improved prompt, "
            "nothing else."
        )

        user_content = f"Improve this image generation prompt: {request.prompt}"
        if request.context:
            user_content += f"\n\nAdditional context: {request.context}"

        logger.debug(f"Refining prompt with model {model}")

        response = self._make_request(
            "POST",
            "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": request.max_tokens,
                "temperature": 0.7,
            },
        )

        data = response.json()
        refined = data["choices"][0]["message"]["content"].strip()

        logger.info(f"Refined prompt: {refined[:100]}...")

        return RefinePromptResponse(
            original_prompt=request.prompt,
            refined_prompt=refined,
            model_used=model,
        )

    # =========================================================================
    # Image Generation
    # =========================================================================

    def generate_image(self, request: GenerateImageRequest) -> bytes:
        """Generate image from text prompt.

        Args:
            request: Generation request with prompt and settings.

        Returns:
            PNG image bytes.

        Raises:
            OpenAIError: On API errors.
        """
        model = request.model or self.config.model

        payload = {
            "model": model,
            "prompt": request.prompt,
            "n": request.n,
            "size": request.size.value,
            "response_format": request.response_format.value,
        }

        if request.quality:
            payload["quality"] = request.quality.value

        if request.style:
            payload["style"] = request.style

        logger.info(f"Generating image with model {model}, prompt: {request.prompt[:50]}...")

        response = self._make_request("POST", "/images/generations", json=payload)
        data = response.json()

        # Extract first image
        image_data = data["data"][0]
        if "b64_json" in image_data:
            return base64.b64decode(image_data["b64_json"])
        elif "url" in image_data:
            # Fetch image from URL
            img_response = httpx.get(image_data["url"], timeout=60)
            return img_response.content
        else:
            raise OpenAIError(
                status="unknown",
                retryable=False,
                detail="No image data in response",
            )

    def generate_image_full(
        self, request: GenerateImageRequest
    ) -> GenerateImageResponse:
        """Generate image(s) and return full response with metadata.

        Args:
            request: Generation request with prompt and settings.

        Returns:
            Full response with all generated images and metadata.

        Raises:
            OpenAIError: On API errors.
        """
        model = request.model or self.config.model

        payload = {
            "model": model,
            "prompt": request.prompt,
            "n": request.n,
            "size": request.size.value,
            "response_format": request.response_format.value,
        }

        if request.quality:
            payload["quality"] = request.quality.value

        if request.style:
            payload["style"] = request.style

        logger.info(f"Generating {request.n} image(s) with model {model}")

        response = self._make_request("POST", "/images/generations", json=payload)
        data = response.json()

        return GenerateImageResponse(
            created=data.get("created", int(time.time())),
            data=[ImageData(**img) for img in data.get("data", [])],
        )

    # =========================================================================
    # Image Editing (Inpaint)
    # =========================================================================

    def edit_image(self, request: EditImageRequest) -> bytes:
        """Edit image with mask (inpaint/outpaint).

        Args:
            request: Edit request with image, mask, and prompt.

        Returns:
            PNG image bytes of edited image.

        Raises:
            OpenAIError: On API errors.
        """
        model = request.model or self.config.model

        # OpenAI expects multipart/form-data for image edits
        files = {
            "image": ("image.png", request.image, "image/png"),
            "mask": ("mask.png", request.mask, "image/png"),
        }

        data = {
            "prompt": request.prompt,
            "n": str(request.n),
            "response_format": request.response_format.value,
        }

        if model:
            data["model"] = model

        if request.size:
            data["size"] = request.size.value

        logger.info(f"Editing image with model {model}, prompt: {request.prompt[:50]}...")

        # For multipart uploads, we need to make the request differently
        # (without the JSON content-type header)
        try:
            response = httpx.post(
                f"{self.config.base_url.rstrip('/')}/images/edits",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                files=files,
                data=data,
                timeout=self.config.timeout_s,
            )
            self._check_response(response)
        except OpenAIError:
            raise
        except httpx.TimeoutException as e:
            raise create_timeout_error(e)
        except httpx.RequestError as e:
            raise create_network_error(e)

        response_data = response.json()

        # Extract first image
        image_data = response_data["data"][0]
        if "b64_json" in image_data:
            return base64.b64decode(image_data["b64_json"])
        elif "url" in image_data:
            img_response = httpx.get(image_data["url"], timeout=60)
            return img_response.content
        else:
            raise OpenAIError(
                status="unknown",
                retryable=False,
                detail="No image data in response",
            )

    def edit_image_full(self, request: EditImageRequest) -> EditImageResponse:
        """Edit image and return full response with metadata.

        Args:
            request: Edit request with image, mask, and prompt.

        Returns:
            Full response with edited image(s) and metadata.

        Raises:
            OpenAIError: On API errors.
        """
        model = request.model or self.config.model

        files = {
            "image": ("image.png", request.image, "image/png"),
            "mask": ("mask.png", request.mask, "image/png"),
        }

        data = {
            "prompt": request.prompt,
            "n": str(request.n),
            "response_format": request.response_format.value,
        }

        if model:
            data["model"] = model

        if request.size:
            data["size"] = request.size.value

        logger.info(f"Editing image with model {model}")

        try:
            response = httpx.post(
                f"{self.config.base_url.rstrip('/')}/images/edits",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                files=files,
                data=data,
                timeout=self.config.timeout_s,
            )
            self._check_response(response)
        except OpenAIError:
            raise
        except httpx.TimeoutException as e:
            raise create_timeout_error(e)
        except httpx.RequestError as e:
            raise create_network_error(e)

        response_data = response.json()

        return EditImageResponse(
            created=response_data.get("created", int(time.time())),
            data=[ImageData(**img) for img in response_data.get("data", [])],
        )

    # =========================================================================
    # Image Variations
    # =========================================================================

    def create_variation(self, request: CreateVariationRequest) -> bytes:
        """Create variation of an image.

        Args:
            request: Variation request with source image.

        Returns:
            PNG image bytes of variation.

        Raises:
            OpenAIError: On API errors.
        """
        model = request.model or self.config.model

        files = {
            "image": ("image.png", request.image, "image/png"),
        }

        data = {
            "n": str(request.n),
            "response_format": request.response_format.value,
        }

        if model:
            data["model"] = model

        if request.size:
            data["size"] = request.size.value

        logger.info(f"Creating image variation with model {model}")

        try:
            response = httpx.post(
                f"{self.config.base_url.rstrip('/')}/images/variations",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                files=files,
                data=data,
                timeout=self.config.timeout_s,
            )
            self._check_response(response)
        except OpenAIError:
            raise
        except httpx.TimeoutException as e:
            raise create_timeout_error(e)
        except httpx.RequestError as e:
            raise create_network_error(e)

        response_data = response.json()

        # Extract first image
        image_data = response_data["data"][0]
        if "b64_json" in image_data:
            return base64.b64decode(image_data["b64_json"])
        elif "url" in image_data:
            img_response = httpx.get(image_data["url"], timeout=60)
            return img_response.content
        else:
            raise OpenAIError(
                status="unknown",
                retryable=False,
                detail="No image data in response",
            )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OpenAICompatClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
