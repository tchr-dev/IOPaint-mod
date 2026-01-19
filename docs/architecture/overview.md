# IOPaint Architecture Overview

High-level overview of the IOPaint system architecture.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Editor    │  │   Header    │  │       Settings         │  │
│  │  (Canvas)   │  │   (Toolbar) │  │    (Model Select)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                │                     │                 │
│         └────────────────┼─────────────────────┘                 │
│                          ▼                                       │
│              ┌─────────────────────┐                             │
│              │  React State (Zustand/Recoil)  │                  │
│              └─────────────────────┘                             │
│                          │                                       │
│                    Socket.IO (Real-time)                        │
└─────────────────────────────────────────────────────────────────┘
                          │ HTTP/API
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   REST API      │  │  Socket.IO      │  │   Model Manager │  │
│  │   Endpoints     │  │   Server        │  │   (Router)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                   │                    │               │
│         └───────────────────┼────────────────────┘               │
│                             ▼                                    │
│              ┌─────────────────────────────┐                     │
│              │   Storage & Budget Guard    │                     │
│              │   (SQLite + File System)    │                     │
│              └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   AI Models           │
              │   (LaMa, SDXL, etc.)  │
              └───────────────────────┘
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `iopaint/` | Main Python backend |
| `iopaint/model/` | Model implementations |
| `iopaint/plugins/` | Plugin system (disabled) |
| `iopaint/openai_compat/` | OpenAI-compatible API |
| `web_app/` | React frontend |
| `docs/` | Documentation |

## Data Flow

1. **Inpainting Request**:
   - User draws mask on image in React frontend
   - Image + mask sent to `/api/v1/inpaint`
   - ModelManager selects and initializes model
   - Model processes image, Socket.IO emits progress
   - Result returned as image response

2. **Model Discovery**:
   - Model classes registered in `iopaint/model/__init__.py`
   - `scan_models()` discovers local models on startup
   - Frontend receives available models via `/api/v1/server_config`
   - User selects model, ModelManager initializes it

## Documentation by Topic

- **Models**: See [Models Architecture](models.md)
- **UI Components**: See [UI Architecture](ui-architecture.md)
- **Storage**: See [Storage Architecture](storage-architecture.md)
- **Vision/Images**: See [Images Vision](images-vision.md)
- **Adding Models**: See AGENTS.md section "Adding New Models"

## Related

- [Development Guide](../guides/development.md)
- [Getting Started Guide](../guides/getting-started.md)
- [AGENTS.md](../../AGENTS.md)
