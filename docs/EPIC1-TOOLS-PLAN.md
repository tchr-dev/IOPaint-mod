# Epic 1.2 Tools Plan

## Scope
Implement the remaining OpenAI-compatible tools:
- Outpaint
- Variations
- Upscale
- Background removal

## Step 1: Review current extension points
- Backend: inspect `iopaint/openai_compat` client, config, and API endpoints in `iopaint/api.py`.
- Frontend: inspect OpenAI UI panels and state in `web_app/src/components/OpenAI/` and `web_app/src/lib/states.ts`.
- Storage/Budget: confirm how requests are recorded and budget guard is applied.

## Step 2: API contracts + backend wiring
- Define request/response schemas per tool.
- Add endpoints (likely `POST /api/v1/openai/outpaint`, `.../variations`, `.../upscale`, `.../background-remove`).
- Update OpenAICompat client to map parameters to provider API.
- Ensure budget guard and job storage (history/images) record each tool type.

### Proposed API Contracts
- `POST /api/v1/openai/edit` (multipart form): `image`, `mask`, `prompt`, optional `n`, `size`, `model`, `response_format`.
- `POST /api/v1/openai/outpaint` (multipart form): same as edit (client expands canvas/mask).
- `POST /api/v1/openai/variations` (multipart form): `image`, optional `n`, `size`, `model`, `response_format`.
- `POST /api/v1/openai/upscale` (multipart form): `image`, optional `scale`, `size`, `model`, `prompt`, `mode`.
- `POST /api/v1/openai/background-remove` (multipart form): `image`, optional `prompt`, `model`, `mode`.

### Tool Modes
- Local models (RealESRGAN/RemoveBG plugins)
- API via specialized prompts (OpenAI-compatible edit with full-image mask)
- API specialized services (stubbed, disabled in UI)

### External Service Stubs (Env Vars)
- `AIE_UPSCALE_SERVICE_URL`, `AIE_UPSCALE_SERVICE_API_KEY`
- `AIE_BG_REMOVE_SERVICE_URL`, `AIE_BG_REMOVE_SERVICE_API_KEY`

## Step 3: Frontend UI + state
- Add UI actions and panels as needed under `web_app/src/components/OpenAI/`.
- Add state/actions in `web_app/src/lib/states.ts` and API calls in `web_app/src/lib/openai-api.ts`.
- Reuse existing cost warnings and history integration.

## Step 4: Tests + docs
- Add tests for endpoints and data validation.
- Update `docs/TODO.md` and any EPIC docs with implemented tool details.
