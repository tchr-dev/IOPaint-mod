# Epic 3 — Storage (SQLite + Files)

> **For AI Assistants**: This document describes the implementation of persistent storage for generation history, images, and audit trail. Use this as reference when working with storage-related code.

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| Pydantic Models | `iopaint/storage/models.py` | Data schemas for jobs and images |
| History Storage | `iopaint/storage/history.py` | SQLite CRUD for generation jobs |
| Image Storage | `iopaint/storage/images.py` | File-based image storage + metadata |
| API Endpoints | `iopaint/api.py` (lines 541-654) | REST API for history/images |
| Frontend Client | `web_app/src/lib/openai-api.ts` (lines 503-779) | TypeScript API functions |
| Frontend State | `web_app/src/lib/states.ts` (lines 1577-1703) | Zustand sync with backend |

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI API    │────▶│   SQLite DB     │
│   (Zustand)     │◀────│   (api.py)       │◀────│   (budget.db)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                          │
                               │                          │
                               ▼                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Image Storage   │────▶│   File System   │
                        │  (images.py)     │     │   (~/.iopaint)  │
                        └──────────────────┘     └─────────────────┘
```

---

## Database Schema

**Location**: Tables created in `iopaint/budget/storage.py` (schema version 2)

### `generation_jobs` Table

```sql
CREATE TABLE generation_jobs (
    id TEXT PRIMARY KEY,               -- UUID
    session_id TEXT NOT NULL,          -- Session identifier
    status TEXT NOT NULL,              -- queued/running/succeeded/failed/blocked_budget
    operation TEXT NOT NULL,           -- generate/edit/refine
    model TEXT NOT NULL,               -- Model used (e.g., "gpt-image-1")
    intent TEXT,                       -- Original user intent
    refined_prompt TEXT,               -- LLM-refined prompt
    negative_prompt TEXT,              -- Negative prompt
    preset TEXT,                       -- draft/final/custom
    params TEXT,                       -- JSON: {size, quality, n}
    fingerprint TEXT,                  -- Deduplication hash
    estimated_cost_usd REAL,           -- Pre-execution estimate
    actual_cost_usd REAL,              -- Post-execution actual
    is_edit INTEGER DEFAULT 0,         -- 1 if edit operation
    error_message TEXT,                -- Error details if failed
    result_image_id TEXT,              -- Reference to images table
    thumbnail_image_id TEXT,           -- Reference to thumbnail
    created_at TIMESTAMP,              -- Job creation time
    completed_at TIMESTAMP             -- Job completion time
);

-- Indexes
CREATE INDEX idx_jobs_session ON generation_jobs(session_id);
CREATE INDEX idx_jobs_fingerprint ON generation_jobs(fingerprint);
CREATE INDEX idx_jobs_created ON generation_jobs(created_at DESC);
CREATE INDEX idx_jobs_status ON generation_jobs(status);
```

### `images` Table

```sql
CREATE TABLE images (
    id TEXT PRIMARY KEY,               -- UUID
    job_id TEXT,                       -- Optional reference to job
    path TEXT NOT NULL,                -- Relative path from data_dir
    thumbnail_path TEXT,               -- Path to thumbnail (lazy-generated)
    width INTEGER,                     -- Image width in pixels
    height INTEGER,                    -- Image height in pixels
    created_at TIMESTAMP,              -- Creation time
    FOREIGN KEY (job_id) REFERENCES generation_jobs(id) ON DELETE SET NULL
);

CREATE INDEX idx_images_job ON images(job_id);
```

### `history_snapshots` Table

```sql
CREATE TABLE history_snapshots (
    id TEXT PRIMARY KEY,               -- UUID
    session_id TEXT NOT NULL,          -- Session identifier
    payload TEXT NOT NULL,             -- JSON snapshot payload
    created_at TIMESTAMP              -- Snapshot creation time
);

CREATE INDEX idx_snapshots_session ON history_snapshots(session_id);
CREATE INDEX idx_snapshots_created ON history_snapshots(created_at DESC);
```

### `models_cache` Table

```sql
CREATE TABLE models_cache (
    provider TEXT PRIMARY KEY,         -- Backend/base_url key
    payload TEXT NOT NULL,             -- JSON model list
    fetched_at TIMESTAMP              -- Cache fetch time
);

CREATE INDEX idx_models_cache_fetched ON models_cache(fetched_at DESC);
```

---

## File Layout

```
~/.iopaint/data/
├── budget.db                    # SQLite database (schema v2)
├── images/
│   ├── {job_id}/                # Images grouped by job
│   │   └── {image_id}.png       # Full resolution images
│   └── {image_id}.png           # Standalone images (no job)
└── thumbnails/
    └── {image_id}.jpg           # 256x256 JPEG thumbnails
```

---

## API Endpoints

### History Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/history` | List jobs (supports `?status=`, `?limit=`, `?offset=`) |
| `POST` | `/api/v1/history` | Create new job entry |
| `GET` | `/api/v1/history/{job_id}` | Get specific job |
| `PATCH` | `/api/v1/history/{job_id}` | Update job (status, cost, etc.) |
| `DELETE` | `/api/v1/history/{job_id}` | Delete job and associated images |
| `DELETE` | `/api/v1/history/clear` | Clear all history for session |

### Snapshot Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/history/snapshots` | List snapshots (supports `?limit=`, `?offset=`) |
| `POST` | `/api/v1/history/snapshots` | Create snapshot (payload JSON) |
| `GET` | `/api/v1/history/snapshots/{snapshot_id}` | Get specific snapshot |
| `DELETE` | `/api/v1/history/snapshots/{snapshot_id}` | Delete snapshot |
| `DELETE` | `/api/v1/history/snapshots/clear` | Clear all snapshots for session |

### Model Cache Endpoints (OpenAI)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/openai/models/cached` | Return cached model list |
| `POST` | `/api/v1/openai/models/refresh` | Refresh cache from provider |

### Image Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/storage/images/{image_id}` | Get full image |
| `GET` | `/api/v1/storage/images/{image_id}/thumbnail` | Get thumbnail (lazy-generated) |

### Request/Response Examples

```bash
# List history
curl -H "X-Session-Id: abc123" http://localhost:8080/api/v1/history?limit=10

# Create job
curl -X POST -H "Content-Type: application/json" \
     -H "X-Session-Id: abc123" \
     http://localhost:8080/api/v1/history \
     -d '{"operation": "generate", "model": "gpt-image-1", "refined_prompt": "A sunset"}'

# Update job status
curl -X PATCH -H "Content-Type: application/json" \
     http://localhost:8080/api/v1/history/{job_id} \
     -d '{"status": "succeeded", "actual_cost_usd": 0.04}'

# Create snapshot
curl -X POST -H "Content-Type: application/json" \
     -H "X-Session-Id: abc123" \
     http://localhost:8080/api/v1/history/snapshots \
     -d '{"payload": {"history": [{"id": "job-1"}]}}'
```

---

## Frontend Integration

### TypeScript API Client

**Location**: `web_app/src/lib/openai-api.ts`

```typescript
// Fetch history from backend
const { jobs, total } = await fetchHistory({ status: "succeeded", limit: 20 })

// Create a job entry
const job = await createHistoryJob({
  operation: "generate",
  model: "gpt-image-1",
  refined_prompt: "A beautiful sunset",
  preset: "draft",
  params: { size: "512x512", quality: "standard", n: 1 },
  estimated_cost_usd: 0.016,
})

// Update job status
await updateHistoryJob(jobId, { status: "succeeded", actual_cost_usd: 0.04 })

// Delete a job
await deleteHistoryJob(jobId)

// Clear all history
await clearHistory()

// Snapshots
const snapshot = await createHistorySnapshot({ history: jobs })
const { snapshots } = await fetchHistorySnapshots({ limit: 20 })

// Get image URLs
const imageUrl = getStoredImageUrl(imageId)
const thumbnailUrl = getStoredThumbnailUrl(imageId)
```

### Zustand State Sync

**Location**: `web_app/src/lib/states.ts`

The history management functions now sync with the backend:

```typescript
// Add to history (syncs to backend automatically)
addToHistory(job)

// Update job (syncs status/cost updates to backend)
updateHistoryJob(id, { status: "succeeded" })

// Remove from history (syncs deletion to backend)
removeFromHistory(id)

// Clear all history (syncs to backend)
clearHistory()

// Explicitly sync with backend (fetch and merge)
await syncHistoryWithBackend()
```

**Sync Strategy**:
- **Write operations**: Fire-and-forget to backend, errors logged but don't block UI
- **Read operations**: `syncHistoryWithBackend()` merges backend data with local state
- **localStorage**: Still used as cache/fallback for offline mode

---

## Implementation Details

### Thread Safety

Both `HistoryStorage` and `ImageStorage` use `threading.local()` for thread-safe database connections, same pattern as `BudgetStorage`.

### Lazy Thumbnails

Thumbnails are generated on first request via `_create_thumbnail()`:
- Source image loaded with PIL
- Converted to RGB (handles PNG transparency)
- Resized to 256x256 max dimension (aspect ratio preserved)
- Saved as 85% quality JPEG

### Schema Migration

The budget storage handles migration from schema version 1 to 2:

```python
# In iopaint/budget/storage.py
SCHEMA_VERSION = 2

def _migrate_schema(self, cursor, from_version):
    if from_version < 2:
        # Create generation_jobs and images tables
        cursor.execute("""CREATE TABLE IF NOT EXISTS generation_jobs (...)""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS images (...)""")
        # Create indexes
        ...
```

### Session Identification

Session ID is extracted from requests in this order:
1. `X-Session-Id` header
2. `session_id` query parameter
3. Fallback: MD5 hash of `client_ip:user_agent`

---

## Code Patterns

### Creating a Job

```python
from iopaint.storage import HistoryStorage, GenerationJobCreate, JobOperation

storage = HistoryStorage(db_path=Path("~/.iopaint/data/budget.db"))

job = storage.save_job(
    session_id="abc123",
    job=GenerationJobCreate(
        operation=JobOperation.GENERATE,
        model="gpt-image-1",
        refined_prompt="A beautiful sunset",
        preset="draft",
        params={"size": "512x512", "quality": "standard", "n": 1},
        estimated_cost_usd=0.016,
    ),
)
print(f"Created job: {job.id}")
```

### Saving an Image

```python
from iopaint.storage import ImageStorage

storage = ImageStorage(
    data_dir=Path("~/.iopaint/data"),
    db_path=Path("~/.iopaint/data/budget.db"),
)

with open("generated.png", "rb") as f:
    image_data = f.read()

record = storage.save_image(image_data, job_id=job.id, format="PNG")
print(f"Saved image: {record.id} at {record.path}")
```

### Listing Jobs

```python
jobs, total = storage.list_jobs(
    session_id="abc123",
    status="succeeded",  # Optional filter
    limit=20,
    offset=0,
)
for job in jobs:
    print(f"{job.id}: {job.refined_prompt[:50]}...")
```

---

## Environment Variables

No new environment variables required. Uses existing `AIE_DATA_DIR` from budget config:

```bash
# Default data directory
AIE_DATA_DIR=~/.iopaint/data

# Model list cache TTL (seconds)
AIE_OPENAI_MODELS_CACHE_TTL_S=3600
```

---

## Testing

### Manual API Test

```bash
# Start server
python3 main.py start --model lama --port 8080

# List history (should return empty initially)
curl http://localhost:8080/api/v1/history

# Create a test job
curl -X POST http://localhost:8080/api/v1/history \
  -H "Content-Type: application/json" \
  -d '{"operation": "generate", "model": "test", "refined_prompt": "test prompt"}'

# List again (should show the new job)
curl http://localhost:8080/api/v1/history
```

### Unit Tests

Create `iopaint/tests/test_storage.py`:

```python
import pytest
from pathlib import Path
import tempfile
from iopaint.storage import HistoryStorage, ImageStorage, GenerationJobCreate, JobOperation

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"

def test_create_and_list_jobs(temp_db):
    storage = HistoryStorage(db_path=temp_db)

    job = storage.save_job(
        session_id="test-session",
        job=GenerationJobCreate(
            operation=JobOperation.GENERATE,
            model="test-model",
            refined_prompt="test prompt",
        ),
    )

    assert job.id is not None
    assert job.status.value == "queued"

    jobs, total = storage.list_jobs("test-session")
    assert total == 1
    assert jobs[0].id == job.id

def test_image_storage(temp_db):
    with tempfile.TemporaryDirectory() as data_dir:
        storage = ImageStorage(
            data_dir=Path(data_dir),
            db_path=temp_db,
        )

        # Create a test PNG (1x1 red pixel)
        import io
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        record = storage.save_image(buffer.getvalue())
        assert record.width == 100
        assert record.height == 100

        # Retrieve
        data = storage.get_image(record.id)
        assert data is not None
```

---

## Notes for AI Assistants

1. **Schema Version**: Always check `SCHEMA_VERSION` in `budget/storage.py` before adding tables
2. **Thread Safety**: Use `threading.local()` pattern for database connections
3. **Fire-and-Forget Sync**: Frontend syncs are non-blocking to keep UI responsive
4. **Lazy Loading**: Thumbnails are generated on first access, not on image save
5. **Session Scope**: History is scoped by session_id for multi-user support
