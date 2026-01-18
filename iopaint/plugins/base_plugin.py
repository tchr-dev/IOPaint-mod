from typing import List, Dict
from loguru import logger
import numpy as np

from iopaint.schema import RunPluginRequest


class BasePlugin:
    name: str
    support_gen_image: bool = False
    support_gen_mask: bool = False

    def __init__(self):
        err_msg = self.check_dep()
        if err_msg:
            logger.error(err_msg)
            exit(-1)

    def gen_image(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        # return RGBA np image or BGR np image
        ...

    def gen_mask(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        # return GRAY or BGR np image, 255 means foreground, 0 means background
        ...

    def check_dep(self):
        ...

    def switch_model(self, new_model_name: str):
        ...

    @property
    def available_models(self) -> List[Dict[str, str]]:
        """Return list of available models for this plugin.

        Returns:
            List of dicts with keys: 'name', 'path', 'url', 'md5'
        """
        return []
