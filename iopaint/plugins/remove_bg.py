import os
import cv2
import numpy as np
from typing import List, Dict
from loguru import logger
import torch
from torch.hub import get_dir

from iopaint.plugins.base_plugin import BasePlugin
from iopaint.schema import Device, RunPluginRequest, RemoveBGModel


def _rmbg_remove(device, *args, **kwargs):
    from rembg import remove

    return remove(*args, **kwargs)


class RemoveBG(BasePlugin):
    name = "RemoveBG"
    support_gen_mask = True
    support_gen_image = True

    def __init__(self, model_name, device):
        super().__init__()
        self.model_name = model_name
        self.device = device

        if model_name.startswith("birefnet"):
            import rembg

            if rembg.__version__ < "2.0.59":
                raise ValueError(
                    "To use birefnet models, please upgrade rembg to >= 2.0.59. pip install -U rembg"
                )

        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")
        os.environ["U2NET_HOME"] = model_dir

        self._init_session(model_name)

    def _init_session(self, model_name: str):
        self.device_warning()

        if model_name == RemoveBGModel.briaai_rmbg_1_4:
            from iopaint.plugins.briarmbg import (
                create_briarmbg_session,
                briarmbg_process,
            )

            self.session = create_briarmbg_session().to(self.device)
            self.remove = briarmbg_process
        elif model_name == RemoveBGModel.briaai_rmbg_2_0:
            from iopaint.plugins.briarmbg2 import (
                create_briarmbg2_session,
                briarmbg2_process,
            )

            self.session = create_briarmbg2_session().to(self.device)
            self.remove = briarmbg2_process
        else:
            from rembg import new_session

            self.session = new_session(model_name=model_name)
            self.remove = _rmbg_remove

    def switch_model(self, new_model_name):
        if self.model_name == new_model_name:
            return

        logger.info(
            f"Switching removebg model from {self.model_name} to {new_model_name}"
        )
        self._init_session(new_model_name)
        self.model_name = new_model_name

    @property
    def available_models(self) -> List[Dict[str, str]]:
        """Return list of available RemoveBG models."""
        models = []
        # Standard rembg models
        rembg_models = [
            "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg",
            "silueta", "isnet-general-use", "birefnet-general",
            "birefnet-general-lite", "birefnet-portrait", "birefnet-dis",
            "birefnet-hrsod", "birefnet-cod", "birefnet-massive"
        ]
        for model_name in rembg_models:
            models.append({
                "name": model_name,
                "path": model_name,
                "url": f"rembg:{model_name}",  # Virtual URL indicating rembg library
                "md5": "",  # MD5 not available for rembg models
            })

        # Bria.ai models
        bria_models = [
            ("briaai/RMBG-1.4", "briaai_rmbg_1_4"),
            ("briaai/RMBG-2.0", "briaai_rmbg_2_0"),
        ]
        for model_path, model_name in bria_models:
            models.append({
                "name": model_name,
                "path": model_path,
                "url": f"bria:{model_path}",  # Virtual URL indicating bria.ai
                "md5": "",  # MD5 not available
            })

        return models

    @torch.inference_mode()
    def gen_image(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)

        # return BGRA image
        output = self.remove(self.device, bgr_np_img, session=self.session)
        return cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)

    @torch.inference_mode()
    def gen_mask(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)

        # return BGR image, 255 means foreground, 0 means background
        output = self.remove(
            self.device, bgr_np_img, session=self.session, only_mask=True
        )
        return output

    def check_dep(self):
        # Plugins disabled - TBD
        pass

    def device_warning(self):
        if self.device == Device.cuda and self.model_name not in [
            RemoveBGModel.briaai_rmbg_1_4,
            RemoveBGModel.briaai_rmbg_2_0,
        ]:
            logger.warning(
                f"remove_bg_device=cuda only supports briaai models({RemoveBGModel.briaai_rmbg_1_4.value}/{RemoveBGModel.briaai_rmbg_2_0.value})"
            )
