# Development Guide

Setting up and working with the IOPaint development environment.

## Prerequisites

- Python 3.12+
- Node.js 18+
- uv (recommended) or pip

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-repo/IOPaint-mod.git
cd IOPaint-mod

# Install Python dependencies
uv sync

# Install frontend dependencies
cd web_app
npm install
cd ..

# Start development servers
./run.sh dev --model lama --port 8080
```

## Development Workflow

### Backend Development

```bash
# Run backend only
uv run python main.py start --model lama --port 8080

# Run tests
pytest iopaint/tests/test_model.py -v

# Run specific test
pytest iopaint/tests/test_model.py::test_lama -v

# Run on CPU only
pytest iopaint/tests/test_model.py -v -k "cpu"
```

### Frontend Development

```bash
cd web_app

# Install dependencies
npm install

# Start dev server (requires backend on port 8080)
npm run dev

# Build for production
npm run build

# Lint (no warnings allowed)
npm run lint
```

### Using the Launcher Script

```bash
./run.sh dev --model lama --port 8080    # Dev mode (backend + Vite)
./run.sh prod --model lama --port 8080   # Production mode (builds, then starts)
./run.sh test                            # Interactive test runner
```

## Configuration

### Environment Variables

Create `web_app/.env.local` to override backend URL:
```
VITE_BACKEND=http://127.0.0.1:8080
```

### Backend Configuration

Use `config/secret.env` for sensitive settings (gitignored):
```bash
AIE_OPENAI_API_KEY=your-key
AIE_OPENAI_BASE_URL=https://api.example.com/v1
AIE_OPENAI_MODEL=gpt-image-1
```

## Model Development

### Adding a New Model

1. Create model class in `iopaint/model/`:
   ```python
   from iopaint.model import InpaintModel

   class MyModel(InpaintModel):
       name = "my-model"
       LAMA_MODEL_URL = "https://..."
       is_erase_model = True

       def init_model(self, device):
           ...

       def forward(self, image, mask, config):
           ...
   ```

2. Register in `iopaint/model/__init__.py:models` dict

3. Add to `DIFFUSERS_MODELS` in `iopaint/const.py` if diffusion-based

4. Update docs/architecture/models.md

### Model Testing

```bash
# Test all models
pytest iopaint/tests/test_model.py -v

# Test specific model
pytest iopaint/tests/test_model.py::test_lama -v

# Test with GPU
pytest iopaint/tests/test_model.py -v -k "cuda"
```

## Code Style

### Python

- Follow PEP 8 with Black formatting
- Use type hints throughout
- Use `loguru` for logging
- See `docs/agents/code-style.md` for details

### TypeScript/React

- Strict TypeScript (`strict: true`)
- Use `cn()` utility for className merging
- Use `forwardRef` for reusable components
- See `docs/agents/code-style.md` for details

## Building for Production

```bash
./run.sh prod --model lama --port 8080
```

This builds the frontend and starts the backend.

## Troubleshooting

- See [Troubleshooting Guide](troubleshooting.md)
- Check backend logs for API errors
- Verify environment variables are set
- Ensure no port conflicts (default: 8080)

## Related

- [Getting Started Guide](getting-started.md)
- [Architecture Overview](architecture/overview.md)
- [AGENTS.md](../AGENTS.md)
