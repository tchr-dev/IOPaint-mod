import cv2
import torch
import numpy as np
from loguru import logger
from iopaint.helper import boxes_from_mask, resize_max_size, pad_img_to_modulo
from iopaint.schema import InpaintRequest, HDStrategy
from iopaint.model.utils.image import match_histograms
from iopaint.model.helper.g_diffuser_bot import expand_image

class InpaintPipeline:
    def __init__(self, model):
        self.model = model

    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB
        masks: [H, W]
        return: BGR IMAGE
        """
        inpaint_result = None
        if config.hd_strategy == HDStrategy.CROP:
            if max(image.shape) > config.hd_strategy_crop_trigger_size:
                logger.info("Run crop strategy")
                boxes = boxes_from_mask(mask)
                crop_result = []
                for box in boxes:
                    crop_image, crop_box = self._run_box(image, mask, box, config)
                    crop_result.append((crop_image, crop_box))

                inpaint_result = image[:, :, ::-1].copy()
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    inpaint_result[y1:y2, x1:x2, :] = crop_image

        elif config.hd_strategy == HDStrategy.RESIZE:
            if max(image.shape) > config.hd_strategy_resize_limit:
                origin_size = image.shape[:2]
                downsize_image = resize_max_size(
                    image, size_limit=config.hd_strategy_resize_limit
                )
                downsize_mask = resize_max_size(
                    mask, size_limit=config.hd_strategy_resize_limit
                )

                logger.info(
                    f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}"
                )
                inpaint_result = self._pad_forward(
                    downsize_image, downsize_mask, config
                )

                # only paste masked area result
                inpaint_result = cv2.resize(
                    inpaint_result,
                    (origin_size[1], origin_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                original_pixel_indices = mask < 127
                inpaint_result[original_pixel_indices] = image[:, :, ::-1][
                    original_pixel_indices
                ]

        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask, config)

        return inpaint_result

    def _pad_forward(self, image, mask, config: InpaintRequest):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.model.pad_mod, square=self.model.pad_to_square, min_size=self.model.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.model.pad_mod, square=self.model.pad_to_square, min_size=self.model.min_size
        )

        image, mask = self.model.forward_pre_process(image, mask, config)

        result = self.model.forward(pad_image, pad_mask, config)
        result = result[0:origin_height, 0:origin_width, :]

        result, image, mask = self.model.forward_post_process(result, image, mask, config)

        if config.sd_keep_unmasked_area:
            mask_expanded = mask[:, :, np.newaxis]
            result = result * (mask_expanded / 255) + image[:, :, ::-1] * (1 - (mask_expanded / 255))
        return result

    def _run_box(self, image, mask, box, config: InpaintRequest):
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)
        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]

    def _crop_box(self, image, mask, box, config: InpaintRequest):
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        if _l < 0:
            r = min(r + abs(_l), img_w)
        if _r > img_w:
            l = max(l - (_r - img_w), 0)
        if _t < 0:
            b = min(b + abs(_t), img_h)
        if _b > img_h:
            t = max(t - (_b - img_h), 0)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        return crop_img, crop_mask, [l, t, r, b]
