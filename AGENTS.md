# Repository Guidelines

## Purpose
- This file guides agentic coding assistants working in this repo.
- Keep changes minimal, focused, and aligned with existing patterns.

## Project Structure
- `iopaint/`: FastAPI backend, CLI, model registry, plugins.
- `iopaint/openai_compat/`: OpenAI-compatible API layer.
- `iopaint/budget/`: Budgeting, rate limiting, and cost tracking.
- `iopaint/storage/`: History, model cache, and image storage.
- `iopaint/tests/`: Pytest suite.
- `web_app/`: React + TypeScript + Vite frontend source.
- `iopaint/web_app/`: Built frontend assets served by backend.
- `assets/`, `docs/`, `scripts/`, `docker/`: support materials.

## Build, Run, Lint, and Test Commands
Backend (Python):
- `uv sync` (preferred) or `pip install -r requirements.txt`.
- `python3 main.py start --model lama --port 8080` runs API server.
- `pytest iopaint/tests/test_model.py -v` runs backend tests.
- Single test by name: `pytest iopaint/tests/test_model.py -k "test_name" -v`.
- Single test file: `pytest iopaint/tests/test_model.py -v`.
- Device-specific subset: `pytest iopaint/tests/test_model.py -k "cpu" -v`.

Frontend (React/TypeScript):
- `cd web_app && npm install` installs frontend deps.
- `npm run dev` starts Vite dev server.
- `npm run build` runs `tsc` and `vite build`.
- `npm run lint` runs ESLint (no warnings allowed).
- `npm run preview` serves a local production build.
- Copy build to backend: `cp -r dist/ ../iopaint/web_app`.

## Tooling and Config Notes
- Python requires 3.12+ (see `pyproject.toml`).
- Frontend uses Vite with `@` alias to `web_app/src`.
- TypeScript strict mode is enabled in `web_app/tsconfig.json`.
- ESLint config lives in `web_app/.eslintrc.cjs`.
- No Python formatter config is present; keep formatting stable.

## Coding Style (Python)
- Follow PEP 8, 4-space indentation.
- Prefer explicit type hints for public functions and models.
- Use `snake_case` for functions/modules and `PascalCase` for classes.
- Keep function bodies small and single-responsibility.
- Use f-strings for string formatting.
- Favor immutable defaults; never use mutable objects as default args.
- Prefer `Path` from `pathlib` for filesystem work.
- Keep FastAPI routes thin; move heavy logic into services/helpers.
- Reuse helpers in `iopaint/helper.py` and `iopaint/model/utils.py`.

## Python Imports and Modules
- Group imports: stdlib, third-party, local; blank line between groups.
- Prefer absolute imports from `iopaint` packages.
- Avoid circular imports; move shared code to `helper` or `services`.
- Remove unused imports and keep import lists minimal.

## Error Handling (Backend)
- Raise `fastapi.HTTPException` for HTTP errors.
- Prefer specific exceptions over bare `except`.
- Only catch broad exceptions at request boundaries.
- When catching exceptions, log context and rethrow if needed.
- Keep error messages user-friendly and consistent.

## Logging and Observability
- Use `loguru.logger` for structured logging.
- Avoid `print` in production paths.
- Include request context (model name, device, job id) when helpful.

## Async and Performance
- Avoid blocking work in request handlers when possible.
- Use async tasks or background helpers for heavy operations.
- Release GPU memory with `torch_gc` where appropriate.

## Coding Style (Frontend)
- Use 2-space indentation and existing formatting.
- Components use `PascalCase`; hooks/variables use `camelCase`.
- Prefer function components and React hooks.
- Keep JSX readable: one prop per line when long.
- Use `@/` alias for imports from `web_app/src`.
- Keep Tailwind class lists consistent with existing ordering.
- Prefer `const` over `let` and avoid `var`.

## TypeScript and Linting
- TypeScript runs in strict mode; address type errors rather than suppressing.
- Avoid `any`; use narrow unions or generics.
- Prefer `unknown` with runtime checks when types are uncertain.
- Use `import type` for type-only imports when needed.
- ESLint warnings are treated as errors; keep `npm run lint` clean.

## Frontend State and API Usage
- Reuse existing state patterns (Recoil/Zustand/React Query) as in nearby code.
- Keep API base URL consistent across services.
- Guard async calls with try/catch and surface errors in UI state.

## Testing Guidance
- Pytest is the only configured test runner.
- Keep new tests under `iopaint/tests/` with `test_*.py` names.
- Prefer testing pure helpers and services instead of heavy model pipelines.
- GPU/CPU model tests may require large downloads; mention this in PRs.

## Formatting and Refactors
- No formatter is enforced for Python; keep diffs minimal.
- Do not reformat entire files unless necessary.
- For frontend, follow ESLint output; no Prettier config is present.
- Prefer small, focused refactors over large rewrites.

## API and CLI Conventions
- FastAPI app wiring lives in `iopaint/api.py`.
- CLI entry points use `typer` in `iopaint/cli.py`.
- Keep CLI flags consistent with existing commands.
- Prefer adding config to existing Pydantic models.

## Plugins and Models
- Plugins live in `iopaint/plugins/` and should use existing base classes.
- Model implementations live in `iopaint/model/`.
- Keep model adapters isolated from API handlers.
- Avoid adding heavy model initialization in module import time.

## Data and Storage
- Model downloads default to `~/.cache/`; override with `--model-dir`.
- History and storage logic lives in `iopaint/storage/`.
- Respect storage interfaces when adding persistence.

## Frontend/Backend Integration
- Backend serves built assets from `iopaint/web_app/`.
- Set `web_app/.env.local` to `VITE_BACKEND=http://127.0.0.1:8080`.
- Keep the backend URL configurable for local development.

## Dependencies and Upgrades
- Keep Python dependencies in `pyproject.toml`.
- Keep frontend deps in `web_app/package.json`.
- Avoid adding new heavy dependencies unless necessary.

## Documentation and Comments
- Update docs only when behavior changes.
- Avoid adding inline comments unless requested or clarifying complex logic.

## Git and PR Notes
- Commit messages: short, imperative summaries.
- PRs should note test commands run or why skipped.
- Add screenshots for UI changes in `web_app/`.

## Agent Notes
- No Cursor rules found (`.cursor/rules/` or `.cursorrules`).
- No GitHub Copilot instructions found (`.github/copilot-instructions.md`).
- If new rules are added later, update this file accordingly.
