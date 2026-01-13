"""InpaintModel adapter for OpenAI-compatible image editing APIs."""

import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from iopaint.model.base import InpaintModel
from iopaint.schema import InpaintRequest

from .client import OpenAICompatClient
from .config import OpenAIConfig
from .models import EditImageRequest, ImageSize
from .errors import OpenAIError


class OpenAICompatModel(InpaintModel):
    """InpaintModel adapter for OpenAI-compatible image editing APIs.

    This model delegates inpainting to OpenAI-compatible APIs (OpenAI, ProxyAPI,
    OpenRouter, etc.) via the /images/edits endpoint.

    The model converts IOPaint's mask format (255 = area to inpaint) to OpenAI's
    format (alpha = 0 = area to edit) and handles size differences between
    input and output images.

    Environment Variables:
        AIE_OPENAI_API_KEY: API key (required)
        AIE_OPENAI_BASE_URL: Base URL for API
        AIE_OPENAI_MODEL: Model for image editing

    Example:
        >>> model = OpenAICompatModel(device="cpu")
        >>> result = model(image, mask, config)
    """

    name = "openai-compat"
    pad_mod = 1  # No padding required for API-based model
    is_erase_model = False  # Requires prompt for editing

    def __init__(self, device, **kwargs):
        """Initialize the OpenAI-compatible model.

        Args:
            device: Device for processing (ignored for API model).
            **kwargs: Additional arguments. Can include:
                - openai_config: OpenAIConfig instance
        """
        self.config: OpenAIConfig = kwargs.get("openai_config") or OpenAIConfig()
        self.client: Optional[OpenAICompatClient] = None
        # Device is stored but not used for API-based model
        self.device = device

        # Initialize client if API key is available
        if self.config.is_enabled:
            self.config.validate()
            self.client = OpenAICompatClient(self.config)
            logger.info(
                f"OpenAI-compatible model initialized with base_url={self.config.base_url}"
            )
        else:
            logger.warning(
                "OpenAI API key not configured. Set AIE_OPENAI_API_KEY to enable."
            )

    def init_model(self, device, **kwargs):
        """Initialize model - handled in __init__."""
        pass

    @staticmethod
    def is_downloaded() -> bool:
        """API-based model doesn't need downloading."""
        return True

    @staticmethod
    def download():
        """No download needed for API-based model."""
        pass

    def _convert_image_to_png_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy RGB image to PNG bytes.

        Args:
            image: [H, W, C] RGB numpy array

        Returns:
            PNG image bytes
        """
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _convert_mask_to_openai_format(self, mask: np.ndarray) -> bytes:
        """Convert IOPaint mask to OpenAI mask format.

        IOPaint mask: 255 = area to inpaint, 0 = area to keep
        OpenAI mask: alpha = 0 (transparent) = area to edit, alpha = 255 = keep

        Args:
            mask: [H, W] grayscale numpy array where 255 = inpaint area

        Returns:
            PNG image bytes with RGBA format for OpenAI
        """
        height, width = mask.shape[:2]

        # Create RGBA image: white with alpha channel
        # Alpha = 0 where we want to edit (mask == 255)
        # Alpha = 255 where we want to keep (mask == 0)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255  # R
        rgba[:, :, 1] = 255  # G
        rgba[:, :, 2] = 255  # B
        rgba[:, :, 3] = 255 - mask  # Alpha: invert mask

        pil_mask = Image.fromarray(rgba, mode="RGBA")
        buffer = io.BytesIO()
        pil_mask.save(buffer, format="PNG")
        return buffer.getvalue()

    def _get_image_size(self, width: int, height: int) -> Optional[ImageSize]:
        """Get appropriate ImageSize for dimensions.

        OpenAI API requires specific sizes. If input doesn't match,
        we'll resize the output back to original dimensions.

        Args:
            width: Image width
            height: Image height

        Returns:
            Best matching ImageSize or None for auto
        """
        # Check if dimensions match a supported size
        size = ImageSize.from_dimensions(width, height)
        if size:
            return size

        # For non-standard sizes, pick the closest larger size
        max_dim = max(width, height)
        if max_dim <= 256:
            return ImageSize.SIZE_256
        elif max_dim <= 512:
            return ImageSize.SIZE_512
        else:
            return ImageSize.SIZE_1024

    def forward(
        self, image: np.ndarray, mask: np.ndarray, config: InpaintRequest
    ) -> np.ndarray:
        """Call OpenAI-compatible edit_image API for inpainting.

        Args:
            image: [H, W, C] RGB numpy array (not normalized, 0-255)
            mask: [H, W] grayscale numpy array (255 = area to inpaint)
            config: InpaintRequest with prompt and other settings

        Returns:
            BGR numpy array result (IOPaint convention)

        Raises:
            OpenAIError: On API errors
            ValueError: If client is not initialized
        """
        if self.client is None:
            raise ValueError(
                "OpenAI client not initialized. "
                "Set AIE_OPENAI_API_KEY environment variable."
            )

        original_height, original_width = image.shape[:2]

        # Convert image and mask to PNG bytes
        image_bytes = self._convert_image_to_png_bytes(image)
        mask_bytes = self._convert_mask_to_openai_format(mask)

        # Determine prompt
        prompt = config.prompt or "Fill in the masked area naturally and seamlessly"

        # Get appropriate size
        size = self._get_image_size(original_width, original_height)

        logger.info(
            f"Calling OpenAI edit_image API: "
            f"size={original_width}x{original_height}, "
            f"api_size={size.value if size else 'auto'}, "
            f"prompt={prompt[:50]}..."
        )

        # Prepare request
        request = EditImageRequest(
            image=image_bytes,
            mask=mask_bytes,
            prompt=prompt,
            n=1,
            size=size,
        )

        # Call API
        try:
            result_bytes = self.client.edit_image(request)
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

        # Convert response to numpy array
        result_pil = Image.open(io.BytesIO(result_bytes))
        result_rgb = np.array(result_pil.convert("RGB"))

        # Resize result to match original input size if different
        if result_rgb.shape[:2] != (original_height, original_width):
            logger.debug(
                f"Resizing result from {result_rgb.shape[:2]} "
                f"to {(original_height, original_width)}"
            )
            result_rgb = cv2.resize(
                result_rgb,
                (original_width, original_height),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Convert RGB to BGR for IOPaint convention
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        logger.info("OpenAI edit_image completed successfully")
        return result_bgr

    def __del__(self):
        """Cleanup client on deletion."""
        if self.client is not None:
            self.client.close()
