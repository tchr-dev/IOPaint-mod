# IOPaint Documentation

A structured documentation system for the IOPaint project.

## Overview

| Directory | Purpose | Contents |
|-----------|---------|----------|
| [guides/](guides/) | How-to guides and tutorials | Getting started, development, troubleshooting |
| [architecture/](architecture/) | System design documents | Models, UI, storage, vision architecture |
| [adr/](adr/) | Architecture Decision Records | Why we made certain design choices |
| [epics/](epics/) | Epic-level planning | Strategic initiatives (3+ weeks) |
| [planning/](planning/) | Task-level planning | Individual features and tasks |
| [agents/](agents/) | AI assistant instructions | Guidance for AI agents working on the codebase |
| [reports/](reports/) | Audit reports and metrics | Security audits, performance reports |

## Quick Links

- **New to IOPaint?** Start with the [Getting Started Guide](guides/getting-started.md)
- **Understanding the codebase?** See [Architecture Overview](architecture/overview.md)
- **Working on a feature?** Check [Planning Dashboard](planning/)
- **Adding a new model?** See [Models Architecture](architecture/models.md)
- **AI assistant?** Read [Agents Guide](agents/)

## Documentation Standards

### Naming Conventions

- Use **lowercase kebab-case**: `ui-architecture.md`, `getting-started.md`
- **No spaces**: Use hyphens instead of spaces
- **Numbered when sequential**: ADRs (`001-topic.md`), Epics (`epic-001-title.md`)
- **Date prefix for reports**: `2026-01-18-audit.md`

### File Formats

- **Guides**: Markdown with clear headings, code examples, and step-by-step instructions
- **Architecture**: Markdown with diagrams, code snippets, and technical depth
- **ADRs**: Use the ADR template from `adr/000-template.md`
- **Epics**: Use the Epic template from `epics/template.md`
- **Planning**: Use frontmatter for metadata (see `planning/template.md`)

## Contributing

When adding documentation:

1. Choose the appropriate directory based on content type
2. Use the correct naming convention
3. Add frontmatter metadata where applicable
4. Update the relevant README.md index if adding new content
5. For AI assistant guidelines, see `agents/README.md`

## Questions?

- **General documentation**: Refer to the appropriate directory's README
- **For AI agents**: See `agents/README.md` or `AGENTS.md` in the project root
