# Epics

Strategic, multi-phase initiatives that span multiple weeks and involve multiple sub-tasks.

## What is an Epic?

An epic is a large body of work that can be broken down into smaller tasks. Epics typically:
- Span 3+ weeks of development
- Involve multiple areas of the codebase
- Have clear success criteria
- Include multiple sub-tasks tracked in planning/

## Structure

```
epics/
├── README.md              # This file - epic tracker dashboard
├── template.md            # Template for new epics
├── active/                # Epics currently in progress
│   └── README.md
├── backlog/               # Planned but not started
│   └── README.md
└── completed/             # Finished epics with reports
    └── README.md
```

## Epic Lifecycle

1. **Backlog**: New epics start here
2. **Active**: Work begins, update status and started date
3. **Completed**: All sub-tasks done, add completion report

## Current Epics

### Active
- None currently active

### Backlog
- None currently in backlog

### Completed
- [Epic 001: Tools Implementation](completed/epic-001-tools.md)
- [Epic 003: Storage Architecture](completed/epic-003-storage.md)
- [Epic 003: Models Cache](completed/epic-003-models-cache.md)
- [Epic 005: Job Runner](completed/epic-005-job-runner.md)

*See individual epic files for full details.*

## Creating a New Epic

1. Copy `template.md` to `epics/backlog/epic-XXX-title.md`
2. Fill in all sections with:
   - Vision and success criteria
   - High-level approach
   - Links to planned sub-tasks in planning/
3. Add to backlog
4. When work begins, move to `active/`
5. When complete, move to `completed/` with report

## Epic Format

See [template.md](template.md) for the full epic template with required sections.
