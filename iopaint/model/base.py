import abc
from typing import List, Optional

import torch
from loguru import logger

from iopaint.helper import (
    switch_mps_device,
)
from iopaint.schema import InpaintRequest
from .utils import get_scheduler


class InpaintModel:
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False
    is_erase_model = False
    supported_devices: List[str] = ["cuda", "mps", "cpu"]

    # Version metadata for update checking
    VERSION: Optional[str] = None
    VERSION_URL: Optional[str] = None

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
        """
        from . import models
        device = switch_mps_device(models, self.name, device)
        self.device = device
        self.init_model(device, **kwargs)
        
        from .pipeline import InpaintPipeline
        self.pipeline = InpaintPipeline(self)

    @abc.abstractmethod
    def init_model(self, device, **kwargs): ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool:
        return False

    @abc.abstractmethod
    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W, 1] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    @staticmethod
    def download(): ...

    @classmethod
    def get_remote_version(cls) -> Optional[str]:
        """Fetch latest version from remote repository."""
        if not cls.VERSION_URL:
            return None

        try:
            import requests
            resp = requests.get(cls.VERSION_URL, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Handle GitHub API format
                if "tag_name" in data:
                    return data["tag_name"].lstrip("v")
                # Handle other API formats if needed
                return data.get("version")
        except Exception as e:
            logger.debug(f"Failed to fetch version for {cls.name}: {e}")
        return None

    @classmethod
    def check_for_updates(cls) -> bool:
        """Check if a newer version is available."""
        if not cls.VERSION or not cls.VERSION_URL:
            return False

        remote_version = cls.get_remote_version()
        if remote_version and remote_version != cls.VERSION:
            logger.info(f"New version available for {cls.name}: {remote_version} (current: {cls.VERSION})")
            return True
        return False

    def forward_pre_process(self, image, mask, config):
        return image, mask

    def forward_post_process(self, result, image, mask, config):
        return result, image, mask

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        return self.pipeline(image, mask, config)


class DiffusionInpaintModel(InpaintModel):
    def __init__(self, device, **kwargs):
        self.model_info = kwargs["model_info"]
        self.model_id_or_path = self.model_info.path
        super().__init__(device, **kwargs)

    def set_scheduler(self, config: InpaintRequest):
        scheduler_config = self.model.scheduler.config
        sd_sampler = config.sd_sampler
        from iopaint.schema import SDSampler
        if config.sd_lcm_lora and self.model_info.support_lcm_lora:
            sd_sampler = SDSampler.lcm
            logger.info(f"LCM Lora enabled, use {sd_sampler} sampler")
        scheduler = get_scheduler(sd_sampler, scheduler_config)
        self.model.scheduler = scheduler

    def forward_pre_process(self, image, mask, config):
        import cv2
        if config.sd_mask_blur != 0:
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return image, mask

    def forward_post_process(self, result, image, mask, config):
        from .utils.image import match_histograms
        if config.sd_match_histograms:
            result = match_histograms(result, image[:, :, ::-1], mask)

        if config.use_extender and config.sd_mask_blur != 0:
            import cv2
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return result, image, mask
