# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

IOPaint is an image inpainting and outpainting tool powered by AI models. It
provides both a web UI and CLI for removing objects, generating content in
masked areas, and enhancing images using various AI models including LaMa,
Stable Diffusion, SDXL, and specialized models like PowerPaint and AnyText. It
also supports OpenAI-compatible APIs (gpt-image-1, dall-e-3) with budget safety
controls.

## Development Commands

### Quick Start

```bash
# Development mode: starts backend + Vite dev server
./run.sh dev --model lama --port 8080

# Production mode: builds frontend, starts backend
./run.sh prod --model lama --port 8080

# Interactive test runner (logs to ./logs/)
./run.sh test
```

### Backend (Python)

```bash
# Install dependencies (using uv - recommended)
uv sync

# Start backend server for development
python3 main.py start --model lama --port 8080

# Start with OpenAI-compatible model
python3 main.py start --model openai-compat --port 8080

# Run tests (requires appropriate device - cuda/mps/cpu)
pytest iopaint/tests/test_model.py -v

# Run a single test
pytest iopaint/tests/test_model.py::test_lama -v

# Run tests for specific device
pytest iopaint/tests/test_model.py -v -k "cpu"
```

### Frontend (React/TypeScript/Vite)

```bash
cd web_app

# Install dependencies
npm install

# Start development server (requires backend running on port 8080)
npm run dev

# Build for production
npm run build

# After build, copy to iopaint package
cp -r dist/ ../iopaint/web_app

# Lint (no warnings allowed)
npm run lint
```

Configure backend URL in `web_app/.env.local`:

```
VITE_BACKEND=http://127.0.0.1:8080
```

## Architecture

### Core Components

**Entry Point**: `main.py` → `iopaint/__init__.py:entry_point()` →
`iopaint/cli.py:typer_app`

**CLI Commands** (`iopaint/cli.py`):

- `start`: Launch web server with FastAPI backend
- `run`: Batch process images from command line
- `download`: Download models from HuggingFace
- `list`: List downloaded models
- `start-web-config`: Launch web-based configuration UI

**API Server** (`iopaint/api.py`):

- FastAPI application with Socket.IO for real-time progress updates
- Serves static React frontend from `iopaint/web_app/`
- Key endpoints: `/api/v1/inpaint`, `/api/v1/model`, `/api/v1/run_plugin_*`

**Model System**:

- `iopaint/model/base.py:InpaintModel`: Abstract base class all models inherit
  from
- `iopaint/model/__init__.py`: Model registry mapping names to classes
- `iopaint/model_manager.py:ModelManager`: Loads/switches models, handles
  ControlNet/BrushNet/PowerPaintV2

**Model Types** (`iopaint/schema.py:ModelType`):

- `INPAINT`: Traditional inpainting models (LaMa, MAT, ZITS, etc.)
- `DIFFUSERS_SD`/`DIFFUSERS_SD_INPAINT`: Stable Diffusion 1.5
- `DIFFUSERS_SDXL`/`DIFFUSERS_SDXL_INPAINT`: Stable Diffusion XL
- `OPENAI_COMPAT`: OpenAI-compatible API models (gpt-image-1, dall-e-3, etc.)

**OpenAI Compatibility** (`iopaint/openai_compat/`):

- `client.py`: Wrapper around OpenAI's image API
- `model_adapter.py`: Adapts OpenAI responses to IOPaint's internal format
- `models.py`: Request/response Pydantic schemas
- `config.py`: OpenAI-specific configuration

**Budget Safety** (`iopaint/budget/`):

- `guard.py:BudgetGuard`: Enforces daily/monthly/session spending caps
- `rate_limiter.py`: Rate limiting between expensive operations
- `cost_estimator.py`: Estimates costs before API calls
- `storage.py`: Persists budget tracking data
- `session.py`: Session-scoped budget tracking

**Plugins** (`iopaint/plugins/`):

- `InteractiveSeg`: SAM-based interactive segmentation for mask generation
- `RemoveBG`: Background removal (briaai models)
- `AnimeSeg`: Anime-specific segmentation
- `RealESRGAN`: Super resolution upscaling
- `GFPGAN`/`RestoreFormer`: Face restoration

**Frontend** (`web_app/src/`):

- React 18 + TypeScript + Vite
- State: Zustand + Recoil
- UI: Radix UI primitives + Tailwind CSS
- API: Axios + TanStack Query

### Request Flow

1. User draws mask on image in React frontend
2. POST to `/api/v1/inpaint` with base64 image + mask
3. `Api.api_inpaint()` decodes images, calls `ModelManager.__call__()`
4. Model processes image, Socket.IO emits progress updates
5. Result returned as image response with seed header

### Adding New Models

1. Create model class inheriting from `InpaintModel` in `iopaint/model/`
2. Implement `init_model()`, `forward()`, `is_downloaded()`, and set `name`
   class attribute
3. Register in `iopaint/model/__init__.py:models` dict
4. Add to `AVAILABLE_MODELS` or `DIFFUSERS_MODELS` in `iopaint/const.py` if
   applicable

### Configuration

- Models download to `~/.cache/` by default (override with `--model-dir`)
- Server config via CLI flags or JSON config file (`--config`)
- Device selection: `cpu`, `cuda`, `mps` (Apple Silicon)
