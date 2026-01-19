# Architecture Decision Records (ADR)

ADRs document significant architectural decisions and the reasoning behind them.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences. ADRs help:

- Track the evolution of architectural choices
- Provide context for new contributors
- Document "why" behind technical decisions
- Avoid repeating past mistakes

## Contents

- [000 - Template](000-template.md) - Use this template for new ADRs
- [001 - Model Modernization](001-model-modernization.md) - Refactoring the model loading system

## ADR Format

Each ADR should include:

1. **Title**: Descriptive title with ADR number
2. **Status**: Proposed, Accepted, Deprecated, or Superseded
3. **Date**: When the decision was made
4. **Context**: The problem or situation that prompted the decision
5. **Decision**: What was decided and why
6. **Consequences**: What becomes easier or harder as a result
7. **Alternatives Considered**: Other options that were evaluated

See [000-template.md](000-template.md) for the full template.

## Creating a New ADR

1. Copy `000-template.md` to a new file with the next sequential number
2. Fill in all sections
3. Set status to "Proposed"
4. Add to the project via PR
5. After review, update status to "Accepted"

## Index

| Number | Title | Status | Date |
|--------|-------|--------|------|
| 001 | Model Modernization | Accepted | 2026-01-18 |

*See individual ADR files for full details.*
