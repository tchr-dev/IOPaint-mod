import os
import time

import cv2
import torch
import torch.nn.functional as F

from iopaint.helper import get_cache_path_by_url, load_jit_model, download_model
from iopaint.schema import InpaintRequest
import numpy as np

from .base import InpaintModel

from .manifest import get_manifest


class ZITS(InpaintModel):
    def __init__(self, device, **kwargs):
        self.manifest = get_manifest("zits")
        self.name = self.manifest.name
        self.is_erase_model = self.manifest.is_erase_model
        self.supported_devices = self.manifest.supported_devices
        self.VERSION = self.manifest.version
        self.VERSION_URL = self.manifest.version_url
        super().__init__(device, **kwargs)
        self.sample_edge_line_iterations = 1

    def init_model(self, device, **kwargs):
        self.wireframe = load_jit_model(
            self.manifest.extra_models["wireframe"]["url"],
            device,
            self.manifest.extra_models["wireframe"]["md5"],
        )
        self.edge_line = load_jit_model(
            self.manifest.extra_models["edge_line"]["url"],
            device,
            self.manifest.extra_models["edge_line"]["md5"],
        )
        self.structure_upsample = load_jit_model(
            self.manifest.extra_models["structure_upsample"]["url"],
            device,
            self.manifest.extra_models["structure_upsample"]["md5"],
        )
        self.inpaint = load_jit_model(
            self.manifest.url, device, self.manifest.md5
        )

    @staticmethod
    def download():
        manifest = get_manifest("zits")
        download_model(
            manifest.extra_models["wireframe"]["url"],
            manifest.extra_models["wireframe"]["md5"],
        )
        download_model(
            manifest.extra_models["edge_line"]["url"],
            manifest.extra_models["edge_line"]["md5"],
        )
        download_model(
            manifest.extra_models["structure_upsample"]["url"],
            manifest.extra_models["structure_upsample"]["md5"],
        )
        download_model(manifest.url, manifest.md5)

    @staticmethod
    def is_downloaded() -> bool:
        manifest = get_manifest("zits")
        model_paths = [
            get_cache_path_by_url(manifest.extra_models["wireframe"]["url"]),
            get_cache_path_by_url(manifest.extra_models["edge_line"]["url"]),
            get_cache_path_by_url(manifest.extra_models["structure_upsample"]["url"]),
            get_cache_path_by_url(manifest.url),
        ]
        return all([os.path.exists(it) for it in model_paths])


    def wireframe_edge_and_line(self, items, enable: bool):
        # 最终向 items 中添加 edge 和 line key
        if not enable:
            items["edge"] = torch.zeros_like(items["masks"])
            items["line"] = torch.zeros_like(items["masks"])
            return

        start = time.time()
        try:
            line_256 = self.wireframe_forward(
                items["img_512"],
                h=256,
                w=256,
                masks=items["mask_512"],
                mask_th=0.85,
            )
        except:
            line_256 = torch.zeros_like(items["mask_256"])

        print(f"wireframe_forward time: {(time.time() - start) * 1000:.2f}ms")

        # np_line = (line[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("line.jpg", np_line)

        start = time.time()
        edge_pred, line_pred = self.sample_edge_line_logits(
            context=[items["img_256"], items["edge_256"], line_256],
            mask=items["mask_256"].clone(),
            iterations=self.sample_edge_line_iterations,
            add_v=0.05,
            mul_v=4,
        )
        print(f"sample_edge_line_logits time: {(time.time() - start) * 1000:.2f}ms")

        # np_edge_pred = (edge_pred[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("edge_pred.jpg", np_edge_pred)
        # np_line_pred = (line_pred[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("line_pred.jpg", np_line_pred)
        # exit()

        input_size = min(items["h"], items["w"])
        if input_size != 256 and input_size > 256:
            while edge_pred.shape[2] < input_size:
                edge_pred = self.structure_upsample(edge_pred)
                edge_pred = torch.sigmoid((edge_pred + 2) * 2)

                line_pred = self.structure_upsample(line_pred)
                line_pred = torch.sigmoid((line_pred + 2) * 2)

            edge_pred = F.interpolate(
                edge_pred,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
            line_pred = F.interpolate(
                line_pred,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )

        # np_edge_pred = (edge_pred[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("edge_pred_upsample.jpg", np_edge_pred)
        # np_line_pred = (line_pred[0][0].numpy() * 255).astype(np.uint8)
        # cv2.imwrite("line_pred_upsample.jpg", np_line_pred)
        # exit()

        items["edge"] = edge_pred.detach()
        items["line"] = line_pred.detach()

    @torch.no_grad()
    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W]
        return: BGR IMAGE
        """
        mask = mask[:, :, 0]
        items = load_image(image, mask, device=self.device)

        self.wireframe_edge_and_line(items, config.zits_wireframe)

        inpainted_image = self.inpaint(
            items["images"],
            items["masks"],
            items["edge"],
            items["line"],
            items["rel_pos"],
            items["direct"],
        )

        inpainted_image = inpainted_image * 255.0
        inpainted_image = (
            inpainted_image.cpu().permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        )
        inpainted_image = inpainted_image[:, :, ::-1]

        # cv2.imwrite("inpainted.jpg", inpainted_image)
        # exit()

        return inpainted_image

    def wireframe_forward(self, images, h, w, masks, mask_th=0.925):
        lcnn_mean = torch.tensor([109.730, 103.832, 98.681]).reshape(1, 3, 1, 1)
        lcnn_std = torch.tensor([22.275, 22.124, 23.229]).reshape(1, 3, 1, 1)
        images = images * 255.0
        # the masks value of lcnn is 127.5
        masked_images = images * (1 - masks) + torch.ones_like(images) * masks * 127.5
        masked_images = (masked_images - lcnn_mean) / lcnn_std

        def to_int(x):
            return tuple(map(int, x))

        lines_tensor = []
        lmap = np.zeros((h, w))

        output_masked = self.wireframe(masked_images)

        output_masked = to_device(output_masked, "cpu")
        if output_masked["num_proposals"] == 0:
            lines_masked = []
            scores_masked = []
        else:
            lines_masked = output_masked["lines_pred"].numpy()
            lines_masked = [
                [line[1] * h, line[0] * w, line[3] * h, line[2] * w]
                for line in lines_masked
            ]
            scores_masked = output_masked["lines_score"].numpy()

        for line, score in zip(lines_masked, scores_masked):
            if score > mask_th:
                try:
                    import skimage

                    rr, cc, value = skimage.draw.line_aa(
                        *to_int(line[0:2]), *to_int(line[2:4])
                    )
                    lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
                except:
                    cv2.line(
                        lmap,
                        to_int(line[0:2][::-1]),
                        to_int(line[2:4][::-1]),
                        (1, 1, 1),
                        1,
                        cv2.LINE_AA,
                    )

        lmap = np.clip(lmap * 255, 0, 255).astype(np.uint8)
        lines_tensor.append(to_tensor(lmap).unsqueeze(0))

        lines_tensor = torch.cat(lines_tensor, dim=0)
        return lines_tensor.detach().to(self.device)

    def sample_edge_line_logits(
        self, context, mask=None, iterations=1, add_v=0, mul_v=4
    ):
        [img, edge, line] = context

        img = img * (1 - mask)
        edge = edge * (1 - mask)
        line = line * (1 - mask)

        for i in range(iterations):
            edge_logits, line_logits = self.edge_line(img, edge, line, masks=mask)

            edge_pred = torch.sigmoid(edge_logits)
            line_pred = torch.sigmoid((line_logits + add_v) * mul_v)
            edge = edge + edge_pred * mask
            edge[edge >= 0.25] = 1
            edge[edge < 0.25] = 0
            line = line + line_pred * mask

            b, _, h, w = edge_pred.shape
            edge_pred = edge_pred.reshape(b, -1, 1)
            line_pred = line_pred.reshape(b, -1, 1)
            mask = mask.reshape(b, -1)

            edge_probs = torch.cat([1 - edge_pred, edge_pred], dim=-1)
            line_probs = torch.cat([1 - line_pred, line_pred], dim=-1)
            edge_probs[:, :, 1] += 0.5
            line_probs[:, :, 1] += 0.5
            edge_max_probs = edge_probs.max(dim=-1)[0] + (1 - mask) * (-100)
            line_max_probs = line_probs.max(dim=-1)[0] + (1 - mask) * (-100)

            indices = torch.sort(
                edge_max_probs + line_max_probs, dim=-1, descending=True
            )[1]

            for ii in range(b):
                keep = int((i + 1) / iterations * torch.sum(mask[ii, ...]))

                assert torch.sum(mask[ii][indices[ii, :keep]]) == keep, "Error!!!"
                mask[ii][indices[ii, :keep]] = 0

            mask = mask.reshape(b, 1, h, w)
            edge = edge * (1 - mask)
            line = line * (1 - mask)

        edge, line = edge.to(torch.float32), line.to(torch.float32)
        return edge, line
