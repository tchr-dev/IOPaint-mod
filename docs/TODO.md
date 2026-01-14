# IOPaint Extension Roadmap

> **For AI Assistants**: This document tracks implementation progress for OpenAI-compatible image generation features. Check status markers (`[x]` = done, `[ ]` = pending) before implementing. Reference commit hashes for context on completed work.

## Quick Status

| Epic | Description | Status | Commit |
|------|-------------|--------|--------|
| Epic 1 | OpenAI-compatible API | ✅ Done | `6478cae` |
| Epic 2 | Budget Safety | ✅ Done | `18f24e3` |
| Epic 3 | Storage (SQLite) | ⏳ Pending | — |
| Epic 4 | UI/UX Components | ✅ Done | `5c26fe6` |
| Epic 5 | Runner/Jobs Queue | ⏳ Pending | — |
| Epic 6 | Testing | ⏳ Pending | — |

---

## Epic 1 — OpenAI-Compatible API Layer

**Goal**: Unified client for OpenAI-compatible providers (OpenAI, ProxyAPI, OpenRouter, etc.)

**Status**: ✅ COMPLETE (`6478cae`)

### E1.1 OpenAI-Compatible Client

- [x] `list_models()` — fetch available models for UI
- [x] `refine_prompt()` — cheap LLM call before expensive generation
- [x] `generate_image()` — text-to-image generation
- [x] `edit_image()` — image+mask inpainting
- [x] Unified error structure (`status`, `retryable`, `detail`)
- [x] Environment config: `AIE_OPENAI_API_KEY`, `AIE_OPENAI_BASE_URL`, `AIE_OPENAI_MODEL`

#### Files Created

```
iopaint/openai_compat/
├── __init__.py          # Package exports
├── config.py            # OpenAIConfig dataclass, env loading
├── errors.py            # OpenAIError, ErrorStatus enum, classify_error()
├── models.py            # Pydantic schemas (EditImageRequest, GenerateImageRequest, etc.)
├── client.py            # OpenAICompatClient with httpx
└── model_adapter.py     # OpenAICompatModel(InpaintModel) adapter
```

#### Files Modified

| File | Changes |
|------|---------|
| `iopaint/schema.py` | Added `ModelType.OPENAI_COMPAT`, updated `need_prompt` property |
| `iopaint/model/__init__.py` | Registered `OpenAICompatModel` in models dict |
| `iopaint/model_manager.py` | Added `openai-compat` to `scan_models()` results |
| `iopaint/api.py` | New endpoints: `GET /api/v1/openai/models`, `POST /api/v1/openai/refine`, `POST /api/v1/openai/generate` |
| `iopaint/cli.py` | CLI flags: `--openai-api-key`, `--openai-base-url`, `--openai-model` |
| `pyproject.toml` | Added `httpx>=0.25.0` dependency |

#### Key Implementation Details

```python
# Mask conversion (IOPaint → OpenAI)
# IOPaint: 255 = area to inpaint
# OpenAI: alpha=0 (transparent) = area to edit
mask_rgba[:, :, 3] = 255 - mask  # Invert to alpha channel

# Result resizing (OpenAI may return different size)
if result.shape[:2] != image.shape[:2]:
    result = cv2.resize(result, (image.shape[1], image.shape[0]))
```

### E1.2 IOPaint Tool Integration

- [x] Inpaint via `openai-compat` model selection
- [ ] Outpaint tool
- [ ] Variations tool
- [ ] Upscale tool
- [ ] Background removal via API

---

## Epic 2 — Budget Safety

**Goal**: Prevent accidental budget burns (double-click, repeated requests, expensive operations without warning)

**Status**: ✅ COMPLETE (`18f24e3`)

### E2.1 BudgetGuard (Hard Caps)

- [x] Daily/monthly/session spending caps
- [x] All paid calls go through BudgetGuard
- [x] `blocked_budget` status with clear UI message

#### Files Created

```
iopaint/budget/
├── __init__.py
├── guard.py             # BudgetGuard class with cap enforcement
├── ledger.py            # Cost tracking and persistence
└── config.py            # BudgetConfig with env loading
```

#### Files Modified

| File | Changes |
|------|---------|
| `iopaint/api.py` | Added `/api/v1/budget/status`, `/api/v1/budget/estimate` endpoints |
| `iopaint/openai_compat/client.py` | Integrated BudgetGuard checks before API calls |

### E2.2 Deduplication by Fingerprint

- [x] Fingerprint = `sha256(normalized(model+action+prompt+params+input_hashes))`
- [x] Return cached result if fingerprint matches recent success
- [x] Configurable cache TTL

### E2.3 Rate Limiting

- [x] 1 expensive operation per N seconds per session
- [x] UI button disable during execution
- [x] Backend fingerprint-based blocking

### E2.4 Cost Awareness UI

- [x] Cost tier display (low/medium/high)
- [x] `estimated_cost_usd` calculation
- [x] Warning modal for high-cost operations

---

## Epic 3 — Storage (SQLite + Files)

**Goal**: Audit trail, job replay, gallery, history restoration, budget ledger persistence

**Status**: ⏳ PENDING

### E3.1 SQLite Schema

- [ ] `jobs` table (id, status, params, timestamps)
- [ ] `images` table (id, job_id, path, metadata)
- [ ] `history_snapshots` table
- [ ] `budget_ledger` table (estimated vs actual costs)
- [ ] `models_cache` table
- [ ] Migration system (alembic or custom)

#### Suggested Schema

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,  -- queued/running/succeeded/failed/blocked_budget
    operation TEXT NOT NULL,  -- generate/edit/refine
    model TEXT NOT NULL,
    prompt TEXT,
    params JSON,
    fingerprint TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE images (
    id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES jobs(id),
    path TEXT NOT NULL,
    thumbnail_path TEXT,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_jobs_fingerprint ON jobs(fingerprint);
CREATE INDEX idx_jobs_created ON jobs(created_at DESC);
```

### E3.2 File Layout

- [ ] `data/images/{job_id}/{image_id}.{ext}` — generated images
- [ ] `data/input/{sha256}.{ext}` — source images (deduped)
- [ ] `data/thumbs/{image_id}.jpg` — thumbnails (optional)

### E3.3 History & Projects

- [ ] History persistence across sessions
- [ ] Project export/import
- [ ] Gap-free history restoration

---

## Epic 4 — UI/UX: Refine → Generate/Edit Flow

**Goal**: Intuitive workflow for prompt refinement and image generation

**Status**: ✅ COMPLETE (`5c26fe6`)

### E4.1 Generation Flow

- [x] Intent input field (raw user idea)
- [x] "Refine Prompt" button (cheap LLM call)
- [x] Editable refined prompt + negative prompt
- [x] Presets: Draft (512, standard), Final (1024, HD), Custom
- [x] Confirmation modal for expensive operations

#### Files Created

```
web_app/src/components/OpenAI/
├── index.tsx              # Component exports
├── IntentInput.tsx        # Raw intent input with refine button
├── PromptEditor.tsx       # Refined + negative prompt editing
├── GenerationPresets.tsx  # Draft/Final/Custom preset selector
├── CostDisplay.tsx        # Cost estimate badge + budget status
├── CostWarningModal.tsx   # High-cost confirmation dialog
├── OpenAIGeneratePanel.tsx # Main generation workflow panel
├── OpenAIEditPanel.tsx    # Edit/inpainting workflow
├── GenerationHistory.tsx  # Job history with filtering
└── HistoryItem.tsx        # Single history entry component

web_app/src/components/ui/
└── badge.tsx              # StatusBadge for job status display

web_app/src/lib/
└── openai-api.ts          # TypeScript API client
```

#### Files Modified

| File | Changes |
|------|---------|
| `web_app/src/components/Header.tsx` | Added Local/OpenAI mode toggle switch |
| `web_app/src/components/SidePanel/index.tsx` | Integrated OpenAI panels based on mode |
| `web_app/src/lib/states.ts` | Added `openAIState` slice with generation workflow state |
| `web_app/src/lib/types.ts` | Added OpenAI types: `GenerationJob`, `GenerationPreset`, `BudgetStatus`, etc. |
| `web_app/package.json` | Added `date-fns` dependency |

#### State Management (Zustand)

```typescript
// web_app/src/lib/states.ts
openAIState: {
  isOpenAIMode: boolean,
  isOpenAIGenerating: boolean,
  openAIIntent: string,
  openAIRefinedPrompt: string,
  openAINegativePrompt: string,
  selectedPreset: GenerationPreset,
  customPresetConfig: PresetConfig,
  generationHistory: GenerationJob[],
  historyFilter: 'all' | 'succeeded' | 'failed',
  budgetStatus: BudgetStatus | null,
  // ... actions
}
```

### E4.2 Edit Flow

- [x] Source image preview
- [x] Mask status indicator
- [x] Edit prompt input
- [x] Apply Edit button with loading state

### E4.3 History/Gallery

- [x] Job list with thumbnail, prompt snippet, status, time
- [x] Filter by status (All/Succeeded/Failed)
- [x] Actions: Open in Editor, Copy Prompt, Re-run, Delete
- [x] Clear all history

---

## Epic 5 — Runner/Jobs Queue

**Goal**: Non-blocking UI with managed job pipeline

**Status**: ⏳ PENDING

### E5.1 Job Runner (Local MVP)

- [ ] Job statuses: `queued` → `running` → `succeeded`/`failed`/`blocked_budget`
- [ ] Progress updates via Socket.IO
- [ ] Audit: save job input/params before API call
- [ ] Retry logic for transient errors (429, 5xx)

#### Suggested Implementation

```python
# iopaint/runner/job_runner.py
class JobRunner:
    def __init__(self, db: Database, budget_guard: BudgetGuard):
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.db = db
        self.budget_guard = budget_guard

    async def submit(self, job: Job) -> str:
        """Submit job to queue, return job_id"""
        await self.db.save_job(job)
        await self.queue.put(job)
        return job.id

    async def run(self):
        """Main runner loop"""
        while True:
            job = await self.queue.get()
            await self._process_job(job)
```

### E5.2 Cancellation

- [ ] Cancellation token in client/runner
- [ ] UI cancel button during generation
- [ ] Graceful cleanup on cancel

---

## Epic 6 — Testing

**Goal**: Lock down contracts, prevent regressions in history/budget

**Status**: ⏳ PENDING

### E6.1 Unit Tests

- [ ] Fingerprint determinism
- [ ] BudgetGuard: block/allow logic
- [ ] Retry logic: 429/timeout retry, 400 no retry
- [ ] Undo/redo for brush and AI edits

#### Test Files to Create

```
iopaint/tests/
├── test_openai_client.py      # Mock HTTP responses
├── test_budget_guard.py       # Cap enforcement
├── test_fingerprint.py        # Determinism checks
└── test_error_classification.py
```

### E6.2 Integration Tests (Mock API)

- [ ] `/v1/images/generations`: file saved, DB records created, dedupe works
- [ ] `/v1/images/edits`: correct mask/params conversion
- [ ] Budget tracking end-to-end

### E6.3 Playwright E2E (Optional)

- [ ] Full flow: load → refine → draft generate → final generate → edit
- [ ] Budget warning modal interaction
- [ ] History actions

---

## Environment Variables Reference

```bash
# OpenAI-Compatible API
AIE_OPENAI_API_KEY=sk-xxx          # Required for openai-compat model
AIE_OPENAI_BASE_URL=https://api.openai.com/v1  # Default
AIE_OPENAI_MODEL=gpt-image-1       # Default model
AIE_OPENAI_TIMEOUT_S=120           # Request timeout

# Budget Safety
AIE_BUDGET_DAILY_CAP=10.0          # USD per day (0 = unlimited)
AIE_BUDGET_MONTHLY_CAP=100.0       # USD per month (0 = unlimited)
AIE_BUDGET_SESSION_CAP=5.0         # USD per session (0 = unlimited)
AIE_RATE_LIMIT_SECONDS=10          # Min seconds between expensive ops
```

---

## CLI Usage

```bash
# Start with OpenAI-compatible model
python3 main.py start --model openai-compat --port 8080

# With explicit config
python3 main.py start \
  --model openai-compat \
  --openai-api-key sk-xxx \
  --openai-base-url https://api.proxyapi.ru/openai/v1

# With budget caps
AIE_BUDGET_DAILY_CAP=5.0 python3 main.py start --model openai-compat
```

---

## API Endpoints Reference

### OpenAI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/openai/models` | List available models |
| POST | `/api/v1/openai/refine` | Refine prompt via LLM |
| POST | `/api/v1/openai/generate` | Generate image from text |
| POST | `/api/v1/inpaint` | Edit image with mask (when model=openai-compat) |

### Budget Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/budget/status` | Get current budget status |
| POST | `/api/v1/budget/estimate` | Estimate operation cost |

---

## Notes for AI Assistants

1. **Before implementing**: Check the status markers above. Don't re-implement completed features.

2. **Code style**: Follow existing patterns in `iopaint/` (Python) and `web_app/src/` (TypeScript/React).

3. **Testing**: Run `npm run build` in `web_app/` to verify frontend compiles. Run `pytest` for backend tests.

4. **Commits**: Use conventional commit messages. Include `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>` for AI-assisted commits.

5. **Dependencies**: Use `uv sync` for Python deps, `npm install` for frontend.
