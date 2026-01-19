# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

IOPaint is an image inpainting/outpainting tool powered by AI models. It provides a
web UI and CLI for removing objects, generating content in masked areas, and enhancing
images using AI models like LaMa, Stable Diffusion, SDXL, PowerPaint, and AnyText. It also
supports OpenAI-compatible APIs (gpt-image-1, dall-e-3) with budget safety controls.

## Development Commands

```bash
./run.sh dev --model lama --port 8080    # Development mode
./run.sh prod --model lama --port 8080   # Production mode
./run.sh test                            # Interactive test runner
```

### Backend (Python)

```bash
uv sync                                                 # Install dependencies
python3 main.py start --model lama --port 8080        # Start server
pytest iopaint/tests/test_model.py -v                   # Run tests
pytest iopaint/tests/test_model.py::test_lama -v        # Single test
pytest iopaint/tests/test_model.py -v -k "cpu"         # Filter by device
```

### Frontend (React/TypeScript/Vite)

```bash
cd web_app
npm install               # Install dependencies
npm run dev               # Start dev server (backend on port 8080)
npm run build             # Build for production
npm run lint              # Lint (no warnings allowed)
```

Configure backend URL in `web_app/.env.local`:
```
VITE_BACKEND=http://127.0.0.1:8080
```

## Code Style Guidelines

### Python (Backend)

**Imports**: Standard library → third-party → local (alphabetical)
```python
import os
from typing import List, Optional
import cv2
import torch
from loguru import logger
from iopaint.schema import InpaintRequest
```

**Types**: Use `typing` module type hints; Pydantic for data models
```python
def forward(self, image, mask, config: InpaintRequest) -> np.ndarray:
    """Input and output images have same size [H, W, C] RGB."""
    ...
```

**Naming**: `snake_case` for functions/variables, `PascalCase` for classes
```python
class LaMa(InpaintModel):
    name = "lama"
    LAMA_MODEL_URL = "https://..."
    def forward(self, image, mask): ...
```

**Error Handling**: Use specific exceptions; log with `loguru`
```python
try:
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200: ...
except Exception as e:
    logger.debug(f"Failed to fetch: {e}")
```

**Abstract Methods**: Use `@abc.abstractmethod` for required implementations

### TypeScript/React (Frontend)

**Imports**: External libraries first, then local with `@/` alias
```typescript
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
```

**Naming**: `camelCase` for variables/functions, `PascalCase` for components
```typescript
const [isOpen, setIsOpen] = useState(false)
export function SettingsDialog() { ... }
```

**Components**: Use `forwardRef` for components accepting ref
```typescript
const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, ...props }, ref) => { ... }
)
Button.displayName = "Button"
export { Button }
```

**Variants**: Use `class-variance-authority (cva)` for component variants
```typescript
const buttonVariants = cva("base-classes", {
  variants: { variant: { default: "...", ghost: "..." } }
})
```

**className Merging**: Always use `cn()` utility (tailwind-merge + clsx)
```typescript
<Comp className={cn(buttonVariants({ size }), "additional-class")} />
```

**Type Safety**: Enable `strict: true` in tsconfig; prefer proper types over `any`

## Architecture

### Entry Point

`main.py` → `iopaint/__init__.py:entry_point()` → `iopaint/cli.py:typer_app`

### CLI Commands (`iopaint/cli.py`)

- `start`: Launch web server with FastAPI backend
- `run`: Batch process images from command line
- `download`: Download models from HuggingFace
- `list`: List downloaded models

### API Server (`iopaint/api.py`)

FastAPI + Socket.IO for real-time progress; serves static React frontend from
`iopaint/web_app/`. Key endpoints: `/api/v1/inpaint`, `/api/v1/model`, `/api/v1/run_plugin_*`

### Model System

- `iopaint/model/base.py:InpaintModel`: Abstract base class
- `iopaint/model/__init__.py`: Model registry
- `iopaint/model_manager.py:ModelManager`: Loads/switches models
- `iopaint/model/pool.py:ModelPool`: Component-level weight sharing

**Model Types** (`iopaint/schema.py:ModelType`):
- `INPAINT`: Traditional (LaMa, MAT, ZITS)
- `DIFFUSERS_SD`/`SD_INPAINT`: Stable Diffusion 1.5
- `DIFFUSERS_SDXL`/`SDXL_INPAINT`: Stable Diffusion XL
- `OPENAI_COMPAT`: OpenAI-compatible API models

### Plugin System (TBD)

Plugins are currently **disabled** to focus on core functionality. The infrastructure is preserved but not initialized:

- `iopaint/plugins/__init__.py`: `build_plugins()` returns empty dict `{}`
- `iopaint/plugins/remove_bg.py`: `check_dep()` validation disabled
- Frontend: All plugin UI components removed (Plugins.tsx deleted)

To re-enable plugins:
1. Restore `build_plugins()` to return initialized plugins
2. Add `PluginName` enum and `PluginParams` interface back to `web_app/src/lib/types.ts`
3. Restore plugin state in `web_app/src/lib/states.ts`
4. Restore plugin components and imports in Editor.tsx, Header.tsx, Workspace.tsx, Settings.tsx
5. Re-enable RemoveBG/RealESRGAN toolbar buttons in Editor.tsx

### OpenAI Compatibility (`iopaint/openai_compat/`)

`client.py`: OpenAI image API wrapper
`model_adapter.py`: Adapts responses to IOPaint format
`budget/`: Daily/monthly/session spending caps

### Frontend (`web_app/src/`)

React 18 + TypeScript + Vite; State via Zustand + Recoil; UI: Radix UI + Tailwind CSS;
API: Axios + TanStack Query

### Request Flow

1. User draws mask on image in React frontend
2. POST to `/api/v1/inpaint` with base64 image + mask
3. `Api.api_inpaint()` decodes images, calls `ModelManager.__call__()`
4. Model processes image, Socket.IO emits progress
5. Result returned as image response with seed header

### Adding New Models

1. Create model class inheriting from `InpaintModel` in `iopaint/model/`
2. Implement `init_model()`, `forward()`, `is_downloaded()`, set `name`
3. Register in `iopaint/model/__init__.py:models` dict
4. For inpaint models: set `is_erase_model = True`
   For diffusion models: add to `DIFFUSERS_MODELS` in `iopaint/const.py`

### Configuration

- Models: `~/.cache/` by default (override with `--model-dir`)
- Server: CLI flags or JSON config file (`--config`)
- Device: `cpu`, `cuda`, `mps` (Apple Silicon)

## Documentation Structure

IOPaint uses a structured documentation system organized by purpose and lifecycle.

### Directory Guide

| Directory | Purpose | When to Use |
|-----------|---------|-------------|
| `docs/guides/` | Getting started, development, troubleshooting | User-facing how-to documentation |
| `docs/architecture/` | System design, technical decisions | Understanding how systems work |
| `docs/adr/` | Architecture Decision Records | Recording why we made design choices |
| `docs/epics/` | Epic-level planning (3+ weeks) | Large initiatives with multiple phases |
| `docs/planning/` | Task-level planning (days to weeks) | Individual features and tasks |
| `docs/agents/` | AI assistant instructions | Agent-specific guidance and rules |
| `docs/reports/` | Audit reports, metrics, analysis | Security audits, performance reports |

### Working with Epics and Planning

**Epics** (`docs/epics/`) - Strategic, large initiatives:
- Use numbered format: `epic-001-title.md`
- Lifecycle: `backlog/` → `active/` → `completed/`
- Contains success criteria and sub-tasks
- Links to individual todos in planning/

**Planning** (`docs/planning/`) - Tactical, actionable tasks:
- Individual features and bug fixes
- Lifecycle: `backlog/` → `active/`
- Completed tasks move to relevant epic or get archived
- Use descriptive kebab-case names

### Workflow for New Work

1. **New Epic**: Create in `docs/epics/backlog/epic-XXX-title.md`
2. **Start Epic**: Move to `active/`, update status and started date
3. **Create Tasks**: Add individual todos to `docs/planning/backlog/`
4. **Work on Task**: Move task to `planning/active/`, implement, complete
5. **Complete Epic**: When all tasks done, move to `epics/completed/` with report

### Architecture Decision Records (ADR)

Use numbered ADR format in `docs/adr/`:
- Format: `001-title.md`, `002-title.md`, etc.
- Use template from `docs/adr/000-template.md`
- Always include: Status, Date, Context, Decision, Consequences
- Link related ADRs when one supersedes another

### File Naming Conventions

- **Lowercase kebab-case**: `ui-architecture.md`, `getting-started.md`
- **No spaces**: Use hyphens instead
- **Numbered when sequential**: ADRs (`001-topic.md`), Epics (`epic-001-title.md`)
- **Date prefix for reports**: `2026-01-18-audit.md`

### Finding Documentation

1. Start with `docs/README.md` for overview and navigation
2. Each major directory has its own `README.md` index
3. Use `docs/planning/README.md` as dashboard for current work
4. Check `docs/epics/README.md` for strategic initiatives
