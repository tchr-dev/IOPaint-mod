# Getting Started with IOPaint

Quick start guide for new users.

## Prerequisites

- Python 3.12+
- Node.js 18+

## Start the Backend

```bash
# from repo root
uv sync
uv run python main.py start --model openai-compat --port 8080
```

Or use the launcher scripts:

```bash
./run.sh dev    # Development mode (backend + Vite dev server)
./run.sh prod   # Production mode (builds frontend, then starts backend)
```

> If you don't use `uv`, run `python -m pip install -r requirements.txt` and then `python main.py start --model openai-compat --port 8080`.

## Start the Frontend

```bash
cd web_app
npm install
npm run dev
```

Optional: Create `web_app/.env.local` to override the backend URL:
```bash
VITE_BACKEND=http://127.0.0.1:8080
```

## Configure OpenAI-Compatible Providers

Set environment variables before starting the backend:

```bash
export AIE_OPENAI_API_KEY="sk-proxyapi-xxx"
export AIE_OPENAI_BASE_URL="https://api.proxyapi.ru/openai/v1"
export AIE_OPENAI_MODEL="gpt-image-1"
```

Or use a secure config file (gitignored):

1. Copy `config/secret.env.example` to `config/secret.env`
2. Add your API keys:
   ```bash
   AIE_OPENAI_API_KEY=sk-proxyapi-xxx
   AIE_OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1
   AIE_OPENAI_MODEL=gpt-image-1
   ```

3. Override path with `AIE_CONFIG_FILE=path/to/env` if needed

To reload config without restarting:
```bash
kill -HUP <pid>
```

## Verification Checklist

- [ ] Select OpenAI-compatible model in the UI
- [ ] Choose provider (`server`, `proxyapi`, `openrouter`) and model
- [ ] Enter a prompt and generate an image
- [ ] Confirm history shows `queued → running → succeeded`

## Next Steps

- See [Development Guide](development.md) for setup instructions
- See [Architecture Overview](../architecture/overview.md) for system design
- See [Troubleshooting](troubleshooting.md) for common issues
