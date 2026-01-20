import os

import cv2
import torch

from iopaint.helper import (
    load_jit_model,
    download_model,
    get_cache_path_by_url,
    boxes_from_mask,
    resize_max_size,
    norm_img,
)
from .base import InpaintModel
from iopaint.schema import InpaintRequest

from .manifest import get_manifest


class MIGAN(InpaintModel):
    def __init__(self, device, **kwargs):
        self.manifest = get_manifest("migan")
        self.name = self.manifest.name
        self.is_erase_model = self.manifest.is_erase_model
        self.supported_devices = self.manifest.supported_devices
        self.VERSION = self.manifest.version
        self.VERSION_URL = self.manifest.version_url
        super().__init__(device, **kwargs)
        self.min_size = 512
        self.pad_mod = 512
        self.pad_to_square = True

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(self.manifest.url, device, self.manifest.md5).eval()

    @staticmethod
    def download():
        manifest = get_manifest("migan")
        download_model(manifest.url, manifest.md5)

    @staticmethod
    def is_downloaded() -> bool:
        manifest = get_manifest("migan")
        return os.path.exists(get_cache_path_by_url(manifest.url))

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        if image.shape[0] == 512 and image.shape[1] == 512:
            return self._pad_forward(image, mask, config)

        boxes = boxes_from_mask(mask)
        crop_result = []
        config.hd_strategy_crop_margin = 128
        for box in boxes:
            crop_image, crop_mask, crop_box = self._crop_box(image, mask, box, config)
            origin_size = crop_image.shape[:2]
            resize_image = resize_max_size(crop_image, size_limit=512)
            resize_mask = resize_max_size(crop_mask, size_limit=512)
            inpaint_result = self._pad_forward(resize_image, resize_mask, config)

            # only paste masked area result
            inpaint_result = cv2.resize(
                inpaint_result,
                (origin_size[1], origin_size[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            original_pixel_indices = crop_mask < 127
            inpaint_result[original_pixel_indices] = crop_image[:, :, ::-1][
                original_pixel_indices
            ]

            crop_result.append((inpaint_result, crop_box))

        inpaint_result = image[:, :, ::-1].copy()
        for crop_image, crop_box in crop_result:
            x1, y1, x2, y2 = crop_box
            inpaint_result[y1:y2, x1:x2, :] = crop_image

        return inpaint_result

    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W] mask area == 255
        return: BGR IMAGE
        """

        image = norm_img(image)  # [0, 1]
        image = image * 2 - 1  # [0, 1] -> [-1, 1]
        mask = (mask > 120) * 255
        mask = norm_img(mask)

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        erased_img = image * (1 - mask)
        input_image = torch.cat([0.5 - mask, erased_img], dim=1)

        output = self.model(input_image)
        output = (
            (output.permute(0, 2, 3, 1) * 127.5 + 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        output = output[0].cpu().numpy()
        cur_res = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return cur_res
