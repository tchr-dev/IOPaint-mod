# Project Architecture Rules (Non-Obvious Only)

- Models are dynamically loaded/switched via `ModelManager` which handles device
  switching and memory cleanup
- Plugin architecture uses factory pattern in `build_plugins()` function with
  lazy initialization
- API server uses FastAPI with Socket.IO mounted at `/ws` for real-time progress
  updates
- History storage uses SQLite with separate tables for jobs, images, and
  snapshots
- Budget system integrates with OpenAI client through decorator pattern for cost
  tracking
- File manager serves images through FastAPI static file routes, not separate
  server
- Image processing pipeline: RGB numpy array → model.forward() → BGR numpy array
  → PNG bytes
- Diffusion models use scaled padding approach with histogram matching for
  quality improvements
- ControlNet/BrushNet integration requires preserving pipeline components during
  switching
- OpenAI compatibility layer adapts external API responses to internal data
  structures
- Job runner implements async queue processing with persistent storage for
  resumability
- Model cache storage uses SQLite to persist downloaded model information
- Plugin models can be switched at runtime without server restart through API
  endpoints
- Image storage automatically generates thumbnails using OpenCV resizing
  operations
