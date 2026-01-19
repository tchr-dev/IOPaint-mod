# IOPaint Development Scripts

Unified development workflow scripts for IOPaint, organized by platform.

## Structure

```
├── run.sh                   # Unix/macOS entry point
├── run.ps1                  # Windows entry point
└── scripts/
    ├── lib/                 # Shared utilities
    │   ├── common.sh       # Bash helpers (Linux/macOS)
    │   └── common.ps1      # PowerShell helpers (Windows)
    ├── unix/                # Linux & macOS command scripts
    │   ├── dev.sh
    │   ├── prod.sh
    │   ├── build.sh
    │   ├── stop.sh
    │   ├── test.sh
    │   ├── jobs.sh
    │   ├── docker.sh
    │   └── publish.sh
    └── windows/             # Windows command scripts
        ├── dev.ps1
        ├── prod.ps1
        ├── build.ps1
        └── test.ps1
```

## Quick Start

### Linux / macOS

```bash
# Start development (backend + frontend)
./run.sh dev

# Build and run production
./run.sh prod

# Run tests
./run.sh test

# Stop servers
./run.sh stop
```

### Windows

```powershell
# Start development (backend + frontend)
.\run.ps1 dev

# Build and run production
.\run.ps1 prod

# Run tests
.\run.ps1 test
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IOPAINT_MODEL` | Model name | `lama` |
| `IOPAINT_PORT` | Backend port | `8080` |
| `IOPAINT_FRONTEND_PORT` | Frontend port hint | `5173` |
| `IOPAINT_VERBOSE` | Enable verbose logging | (empty) |

## Commands

| Command | Unix | Windows | Description |
|---------|------|---------|-------------|
| `dev` | ✓ | ✓ | Start backend + Vite dev server |
| `prod` | ✓ | ✓ | Build frontend, copy assets, start backend |
| `build` | ✓ | ✓ | Build frontend only |
| `stop` | ✓ | ✗ | Stop servers by port |
| `test` | ✓ | ✓ | Run tests (pytest/npm) |
| `docker` | ✓ | ✗ | Build Docker images |
| `publish` | ✓ | ✗ | Build for PyPI |

## Requirements

### Unix
- `bash` 4.0+
- `uv` (Python package manager)
- `node` + `npm`
- `docker` (for `docker` command)

### Windows
- PowerShell 7+
- `uv` in PATH
- Node.js with npm
