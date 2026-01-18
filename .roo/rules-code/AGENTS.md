# Project Coding Rules (Non-Obvious Only)

- Always use `torch_gc()` from `iopaint/model/utils.py` after heavy GPU
  operations to prevent memory leaks
- Model switching requires deleting the old model instance and calling
  `torch_gc()` before initializing the new model
- All models must inherit from `InpaintModel` in `iopaint/model/base.py` and
  implement `init_model()`, `forward()`, and `is_downloaded()`
- Use `switch_mps_device()` from `iopaint/helper.py` when handling MPS device
  switching on macOS
- Image data flows as RGB numpy arrays internally, but models return BGR images
- Mask images are grayscale with 255 representing the area to inpaint
- All API routes should use the `@torch.inference_mode()` decorator for
  performance
- Use `diffuser_callback` for progress updates during diffusion model operations
- When implementing new plugins, inherit from `BasePlugin` in
  `iopaint/plugins/base_plugin.py`
- Always check `is_local_files_only()` from `iopaint/model/utils.py` when
  downloading models
- Use `loguru.logger` for all logging instead of Python's built-in logging
  module
- All Pydantic models should use `model_dump()` instead of deprecated `dict()`
  method
- When adding new API endpoints, follow the pattern of using `add_api_route` in
  the Api class constructor
