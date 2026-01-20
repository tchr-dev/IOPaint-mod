import os
from typing import List

import PIL.Image
import cv2
import torch
from diffusers import AutoencoderKL
from loguru import logger

from iopaint.schema import InpaintRequest, ModelType

from .base import DiffusionInpaintModel
from .helper.cpu_text_encoder import CPUTextEncoderWrapper
from .original_sd_configs import get_config_files
from .utils import (
    handle_from_pretrained_exceptions,
    get_torch_dtype,
    enable_low_mem,
    is_local_files_only,
)


class SDXL(DiffusionInpaintModel):
    name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    pad_mod = 8
    min_size = 512
    model_id_or_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    def __init__(self, device, **kwargs):
        from iopaint.config import get_config
        self.lcm_lora_id = get_config().lcm_lora_models.get("sdxl", "latent-consistency/lcm-lora-sdxl")
        super().__init__(device, **kwargs)

    @classmethod
    def get_shared_components(cls) -> List[str]:
        """Get list of component types that can be shared for SDXL models."""
        return [
            "vae_sdxl",
            "text_encoder_sdxl",
        ]

    @classmethod
    def get_used_components(cls) -> List[str]:
        """Get list of shared components used by this model instance."""
        return cls.get_shared_components()

    def init_model(self, device: torch.device, **kwargs):
        from diffusers.pipelines import StableDiffusionXLInpaintPipeline

        use_gpu, torch_dtype = get_torch_dtype(device, kwargs.get("no_half", False))

        if self.model_info.model_type == ModelType.DIFFUSERS_SDXL:
            num_in_channels = 4
        else:
            num_in_channels = 9

        model_kwargs = {
            **kwargs.get("pipe_components", {}),
            "local_files_only": is_local_files_only(**kwargs),
        }

        # Check for shared components
        shared_components = kwargs.get("shared_components", {})
        if shared_components:
            logger.info(f"Using shared components for SDXL model: {list(shared_components.keys())}")
            # Merge shared components into model_kwargs
            for comp_name, component in shared_components.items():
                if comp_name == "vae_sdxl" and "vae" not in model_kwargs:
                    model_kwargs["vae"] = component
                elif comp_name == "text_encoder_sdxl" and isinstance(component, dict):
                    # SDXL has dual text encoders
                    if "text_encoder" not in model_kwargs:
                        model_kwargs["text_encoder"] = component.get("text_encoder")
                    if "text_encoder_2" not in model_kwargs:
                        model_kwargs["text_encoder_2"] = component.get("text_encoder_2")

        if os.path.isfile(self.model_id_or_path):
            self.model = StableDiffusionXLInpaintPipeline.from_single_file(
                self.model_id_or_path,
                torch_dtype=torch_dtype,
                num_in_channels=num_in_channels,
                load_safety_checker=False,
                original_config_file=get_config_files()['xl'],
                **model_kwargs,
            )
        else:
            if "vae" not in model_kwargs:
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
                )
                model_kwargs["vae"] = vae
            self.model = handle_from_pretrained_exceptions(
                StableDiffusionXLInpaintPipeline.from_pretrained,
                pretrained_model_name_or_path=self.model_id_or_path,
                torch_dtype=torch_dtype,
                variant="fp16",
                **model_kwargs
            )

        enable_low_mem(self.model, kwargs.get("low_mem", False))

        if kwargs.get("cpu_offload", False) and use_gpu:
            logger.info("Enable sequential cpu offload")
            self.model.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.model = self.model.to(device)
            if kwargs["sd_cpu_textencoder"]:
                logger.info("Run Stable Diffusion TextEncoder on CPU")
                self.model.text_encoder = CPUTextEncoderWrapper(
                    self.model.text_encoder, torch_dtype
                )
                self.model.text_encoder_2 = CPUTextEncoderWrapper(
                    self.model.text_encoder_2, torch_dtype
                )

        self.callback = kwargs.pop("callback", None)

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        self.set_scheduler(config)

        img_h, img_w = image.shape[:2]

        output = self.model(
            image=PIL.Image.fromarray(image),
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            num_inference_steps=config.sd_steps,
            strength=0.999 if config.sd_strength == 1.0 else config.sd_strength,
            guidance_scale=config.sd_guidance_scale,
            output_type="np",
            callback_on_step_end=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output
