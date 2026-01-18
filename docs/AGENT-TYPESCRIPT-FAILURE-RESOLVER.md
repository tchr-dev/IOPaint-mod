# AI Agent Instruction: Build Failure Resolver

## Role

You are a senior TypeScript + React build engineer.

## Objective

Make `tsc && vite build` pass in **prod mode** with minimal, correct changes. Do
NOT refactor unless strictly required for type correctness.

## Scope

- React + TypeScript
- Vite
- Zustand + Immer
- Vitest / Testing Library
- OpenAI-compatible job flow logic

## Hard Rules

- Fix **type definitions first**, not tests, unless tests are incorrect.
- Prefer **type-safe solutions** over `any`.
- Do NOT change runtime behavior unless unavoidable.
- Tests must compile; test logic may be adjusted only for typing correctness.
- DOM objects must NOT be stored inside Immer state.

## Required Fixes

### 1. React `act` Usage (Tests)

- Do NOT import `act` from `react`.
- Use one of:
  - Testing Library async helpers (`waitFor`, `findBy*`)
  - `react-dom/test-utils` if absolutely necessary.
- Set `IS_REACT_ACT_ENVIRONMENT` using a safe global cast.

### 2. Global Typings

- Avoid implicit `any` on `globalThis`.
- Use explicit casts or global `.d.ts` declarations.

### 3. BudgetStatus Typing

- `status` must be a strict union:
  - `"ok" | "warning" | "blocked"`
- Test mocks must conform to the union.
- Do NOT widen the union to `string`.

### 4. Job Cost Fields

- `actual_cost_usd` MUST allow numeric values when a job is completed.
- Use `number | undefined` (or `number | null`) consistently.
- Tests and types must agree.

### 5. Job Error Fields

- If jobs can fail, the job type MUST include:
  - `error_message?: string`
- Do NOT use extra properties not defined in the type.

### 6. Immer + DOM Objects (Critical)

- Never store `HTMLImageElement` or other DOM nodes in state.
- Replace with serializable render metadata:
  - `id`, `src`, `width`, `height`, etc.
- DOM nodes must live in components via refs, not in state.

### 7. Zustand `get / set` Compatibility

- `startOpenAIJobPolling` MUST accept the exact `get` / `set` signatures used by
  the store.
- Account for Immer middleware (`WritableDraft`).
- Use generic or compatible function typing—no unsafe casts.

## Completion Criteria

- `./run.sh prod` succeeds.
- `tsc` reports zero errors.
- No new `any` types introduced unless explicitly justified.
- State remains serializable and Immer-safe.

## Output

- Apply code changes directly.
- Keep diffs minimal.
- Do NOT add commentary or explanations.
