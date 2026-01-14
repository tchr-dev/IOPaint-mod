# Repository Guidelines

## Project Structure & Module Organization
- `iopaint/`: Python backend package (FastAPI server, CLI, model registry, plugins, OpenAI-compatible API).
- `iopaint/tests/`: Backend tests (pytest).
- `web_app/`: React + TypeScript + Vite frontend source.
- `iopaint/web_app/`: Built frontend assets served by the backend.
- `assets/`, `docs/`, `scripts/`, `docker/`: Supporting assets, documentation, utilities, and container setup.

## Build, Test, and Development Commands
Backend (Python):
- `uv sync` or `pip install -r requirements.txt`: install backend dependencies.
- `python3 main.py start --model lama --port 8080`: run the API server.
- `pytest iopaint/tests/test_model.py -v`: run backend tests (requires cpu/cuda/mps).

Frontend (React/TypeScript):
- `cd web_app && npm install`: install frontend dependencies.
- `npm run dev`: start Vite dev server (expects backend at `http://127.0.0.1:8080`).
- `npm run build`: build production assets.
- `cp -r dist/ ../iopaint/web_app`: copy built assets into the backend package.
- `npm run lint`: run ESLint checks.

## Coding Style & Naming Conventions
- Python: follow PEP 8, 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- TypeScript/React: 2-space indentation, `PascalCase` component names, `camelCase` variables.
- Use ESLint for frontend style enforcement (`npm run lint`).

## Testing Guidelines
- Framework: pytest for backend tests (`iopaint/tests/`).
- Naming: `test_*.py` and `test_*` functions (see `iopaint/tests/test_model.py`).
- Device-specific runs: `pytest iopaint/tests/test_model.py -v -k "cpu"`.

## Commit & Pull Request Guidelines
- Commit messages use short, imperative summaries (e.g., "Add OpenAI-compatible API support").
- PRs should describe the change, include test results or rationale for skipping tests, and add screenshots for UI changes in `web_app/`.

## Configuration Tips
- Frontend backend URL: set `web_app/.env.local` with `VITE_BACKEND=http://127.0.0.1:8080`.
- Models download to `~/.cache/` by default; override with `--model-dir`.
