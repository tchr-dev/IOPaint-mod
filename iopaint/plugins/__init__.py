from typing import Dict

from loguru import logger

from .anime_seg import AnimeSeg
from .gfpgan_plugin import GFPGANPlugin
from .interactive_seg import InteractiveSeg
from .realesrgan import RealESRGANUpscaler
from .remove_bg import RemoveBG
from .restoreformer import RestoreFormerPlugin
from ..schema import InteractiveSegModel, Device, RealESRGANModel


def build_plugins(
    enable_interactive_seg: bool,
    interactive_seg_model: InteractiveSegModel,
    interactive_seg_device: Device,
    enable_remove_bg: bool,
    remove_bg_device: Device,
    remove_bg_model: str,
    enable_anime_seg: bool,
    enable_realesrgan: bool,
    realesrgan_device: Device,
    realesrgan_model: RealESRGANModel,
    enable_gfpgan: bool,
    gfpgan_device: Device,
    enable_restoreformer: bool,
    restoreformer_device: Device,
    no_half: bool,
) -> Dict:
    # Plugins disabled - TBD
    return {}
