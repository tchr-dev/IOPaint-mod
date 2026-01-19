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
│  ┌─────────────────┐  └─────────────────┘  └─────────────────┘  │
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
| `web_app/` | React frontend |
| `docs/` | Documentation |

## Data Flow
...
## Documentation by Topic

- **Models**: See [Models Architecture](models.md)
- **UI Components**: See [UI Architecture](ui.md)
- **Adding Models**: See AGENTS.md section "Adding New Models"


## Related

- [Development Guide](../guides/development.md)
- [Getting Started Guide](../guides/getting-started.md)
- [AGENTS.md](../../AGENTS.md)
