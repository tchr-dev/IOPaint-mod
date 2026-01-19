# Epic 5 — Job Runner & Queue Plan

## Goals
- Enable async OpenAI-compatible job queue with persisted inputs.
- Add polling-based status updates and cancel UI.
- Preserve inputs on disk for project-style replay (manual cleanup).

## Storage & Persistence
- Store inputs under `data/inputs/{job_id}/`.
- Filenames: `{sha256}.{ext}` where hash is computed from bytes.
- Persist input metadata in `generation_jobs.params`:
  - `input_images`: array of `{type, path, sha256, mime}`
  - `mask_images`: array of `{type, path, sha256, mime}`
- No SQLite blobs.

## Backend Work
1. **Input storage utility**
   - New helper in `iopaint/storage/inputs.py`.
   - Functions: `save_input_bytes`, `load_input_bytes`, `get_input_path`.
2. **API submit flow**
   - In `/api/v1/openai/jobs`, save inputs before enqueue.
   - Persist references in `GenerationJob.params`.
3. **JobRunner updates**
   - Load persisted inputs from disk instead of raw request bytes.
   - Ensure cancellation respects status transitions.
4. **Endpoints**
   - `/api/v1/openai/jobs` submit → queued.
   - `/api/v1/openai/jobs/{job_id}` get status.
   - `/api/v1/openai/jobs/{job_id}/cancel` cancel.
   - No cleanup endpoints for MVP.

## Frontend Work
1. **API client**
   - Add job submit/get/cancel functions in `web_app/src/lib/openai-api.ts`.
2. **State & polling**
   - Update `openAIState` to submit jobs instead of direct generate.
   - Poll current job every 5s while in-progress.
   - Update history entries from backend status.
3. **UI**
   - Add cancel button when job is `queued` or `running`.

## Tests
- Extend `iopaint/tests/test_openai_tools_api.py`:
  - Submit job with input → stored in `data/inputs`.
  - Status polling returns persisted job.
  - Cancel updates status to `cancelled`.

## Docs
- Update Epic 5 items in `docs/TODO.md` once implemented.
