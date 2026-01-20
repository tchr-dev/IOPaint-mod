import os
from typing import List

import cv2
import numpy as np
import torch

from iopaint.helper import (
    norm_img,
    get_cache_path_by_url,
    load_jit_model,
    download_model,
)
from iopaint.schema import InpaintRequest
from .base import InpaintModel
from .manifest import get_manifest


class LaMa(InpaintModel):
    def __init__(self, device, **kwargs):
        self.manifest = get_manifest("lama")
        self.name = self.manifest.name
        self.is_erase_model = self.manifest.is_erase_model
        self.supported_devices = self.manifest.supported_devices
        self.VERSION = self.manifest.version
        self.VERSION_URL = self.manifest.version_url
        super().__init__(device, **kwargs)

    @staticmethod
    def download():
        manifest = get_manifest("lama")
        download_model(manifest.url, manifest.md5)

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(self.manifest.url, device, self.manifest.md5).eval()

    @staticmethod
    def is_downloaded() -> bool:
        manifest = get_manifest("lama")
        return os.path.exists(get_cache_path_by_url(manifest.url))

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res


class AnimeLaMa(LaMa):
    def __init__(self, device, **kwargs):
        self.manifest = get_manifest("anime-lama")
        self.name = self.manifest.name
        self.is_erase_model = self.manifest.is_erase_model
        self.supported_devices = self.manifest.supported_devices
        self.VERSION = self.manifest.version
        self.VERSION_URL = self.manifest.version_url
        # Skip LaMa.__init__ and call InpaintModel.__init__ directly or let super() handle it
        # Actually LaMa.__init__ does what we need but with "lama" manifest.
        # So we just need to set our manifest before calling super.
        super(LaMa, self).__init__(device, **kwargs)

    @staticmethod
    def download():
        manifest = get_manifest("anime-lama")
        download_model(manifest.url, manifest.md5)

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(
            self.manifest.url, device, self.manifest.md5
        ).eval()

    @staticmethod
    def is_downloaded() -> bool:
        manifest = get_manifest("anime-lama")
        return os.path.exists(get_cache_path_by_url(manifest.url))
