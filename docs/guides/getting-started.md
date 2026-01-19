# Getting Started with IOPaint

Quick start guide for new users.

## Prerequisites

- Python 3.12+
- Node.js 18+

## Start the Backend

```bash
# from repo root
uv sync
uv run python main.py start --model lama --port 8080
```

Or use the launcher scripts:

```bash
./run.sh dev    # Development mode (backend + Vite dev server)
./run.sh prod   # Production mode (builds frontend, then starts backend)
```

> If you don't use `uv`, run `python -m pip install -r requirements.txt` and then `python main.py start --model lama --port 8080`.

## Start the Frontend
...
## Next Steps


- See [Development Guide](development.md) for setup instructions
- See [Architecture Overview](../architecture/overview.md) for system design
- See [Troubleshooting](troubleshooting.md) for common issues
