# Project Debug Rules (Non-Obvious Only)

- Websocket progress updates use Socket.IO on `/ws` endpoint, not standard
  WebSocket
- Model loading failures often manifest as CUDA memory errors - check
  `nvidia-smi` during debugging
- On macOS, MPS device issues can cause silent failures without error messages
- Image format conversion issues (RGB/BGR) often cause color distortion in
  output
- Mask preprocessing issues (thresholding, resizing) can cause inpainting to
  fail silently
- When debugging diffusion models, check that scheduler configuration matches
  model expectations
- Memory leaks often occur when model instances aren't properly deleted during
  switching
- GPU memory fragmentation can cause out-of-memory errors even with sufficient
  total memory
- Logging configuration is set in `iopaint/__init__.py` - verbose mode enabled
  with `IOPAINT_VERBOSE=1`
- Test images are stored in `iopaint/tests/` with expected results in
  `iopaint/tests/result/`
- When tests fail, check that model files are properly downloaded to
  `~/.cache/iopaint/`
- File manager issues with input/output directories can cause 404 errors in the
  web UI
- History storage uses SQLite database in `~/.iopaint/data/iopaint.db` by
  default
- Budget tracking data is also stored in the same SQLite database
