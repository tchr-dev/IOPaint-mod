# IOPaint-Mod: OpenAI-Compatible Start Guide

## Prerequisites
- Python 3.12+
- Node.js 18+

## Start the Backend (OpenAI-Compatible Model)

```bash
# from repo root
uv sync
uv run python main.py start --model openai-compat --port 8080
```

Friendly launcher (optional):

```bash
./launch.sh dev
```

Production launcher (builds frontend then starts backend):

```bash
./launch.sh prod
```

> If you don’t use `uv`, run `python -m pip install -r requirements.txt` and then
> `python main.py start --model openai-compat --port 8080`.

## Start the Frontend (Vite)

```bash
cd web_app
npm install
npm run dev
```

Optional local override (create `web_app/.env.local`):

```bash
VITE_BACKEND=http://127.0.0.1:8080
```

## Configure ProxyAPI.ru (OpenAI-Compatible)

Set these environment variables before starting the backend:

```bash
export AIE_OPENAI_API_KEY="sk-proxyapi-xxx"
export AIE_OPENAI_BASE_URL="https://api.proxyapi.ru/openai/v1"
export AIE_OPENAI_MODEL="gpt-image-1"
```

Then start the backend:

```bash
uv run python main.py start --model openai-compat --port 8080
```

In the UI, set the provider to `proxyapi` (if available). If a base URL field is shown, use:

```
https://api.proxyapi.ru/openai/v1
```

## Secure Config File (Hot Reload)

Create `config/secret.env` (gitignored) with `.env`-style entries:

```bash
AIE_OPENAI_API_KEY=sk-proxyapi-xxx
AIE_OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1
AIE_OPENAI_MODEL=gpt-image-1
```

You can start from the template at `config/secret.env.example`.

OS environment variables still take precedence over the file. Override path with `AIE_CONFIG_FILE=path/to/env`.

To reload on the fly:

```bash
kill -HUP <pid>
```

The backend also exposes safe config keys at `GET /api/v1/config/public`.

---

# Manual Testing Checklist (OpenAI-Compatible)

## A) Core Generation Flow
- [ ] Select OpenAI-compatible model in the UI.
- [ ] Choose provider (`server`, `proxyapi`, `openrouter`) and model.
- [ ] Enter a prompt and generate an image.
- [ ] Confirm history shows `queued → running → succeeded`.

## B) Job Queue + Polling
- [ ] Trigger generation and observe status transitions.
- [ ] Verify status updates without refresh.
- [ ] Confirm completed job shows image thumbnail/result.

## C) Cancel Flow
- [ ] Start a generation and click **Cancel** while queued/running.
- [ ] Confirm status becomes `cancelled` in history.
- [ ] Verify polling stops for the cancelled job.

## D) History Filters
- [ ] Filter history to **Failed/Cancelled**.
- [ ] Confirm cancelled entries appear in the failed filter list.
- [ ] Switch to **All** and confirm everything shows again.

## E) Edits (Inpainting)
- [ ] Load an image, draw a mask, enter a prompt.
- [ ] Run edit and confirm result image appears.
- [ ] Confirm history marks it as an edit.

## F) Variations / Upscale / Background Remove
- [ ] Run variations.
- [ ] Run upscale.
- [ ] Run background remove.
- [ ] Confirm each adds history entries and images render.

## G) Budget/Cost UI (if enabled)
- [ ] Confirm cost estimate appears before generation.
- [ ] If budget caps are set, verify blocked/caution behavior.

---

## Troubleshooting
- If models list is empty, confirm `AIE_OPENAI_API_KEY` and base URL.
- If requests fail, check backend logs for OpenAI errors.
- If UI shows `blocked_budget`, inspect budget env vars or disable caps.
