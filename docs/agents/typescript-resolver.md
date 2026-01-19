# Build Failure Resolver

Guidelines for resolving TypeScript/React build failures.

## Objective

Make `tsc && vite build` pass with minimal, correct changes. Do NOT refactor unless strictly required for type correctness.

## Scope

- React + TypeScript
- Vite
- Zustand + Immer
- Vitest / Testing Library
- OpenAI-compatible job flow logic

## Hard Rules

1. Fix **type definitions first**, not tests, unless tests are incorrect
2. Prefer **type-safe solutions** over `any`
3. Do NOT change runtime behavior unless unavoidable
4. Tests must compile; adjust test logic only for typing correctness
5. **DOM objects must NOT be stored inside Immer state**

## Common Fixes

### React `act` Usage in Tests

**Problem**: `Module '"react"' has no exported member 'act'`

**Solution**:
```typescript
import { act } from "react/test-utils"
```

### GlobalThis Index Signature

**Problem**: `Element implicitly has an 'any' type` for globalThis

**Solution**:
```typescript
declare global {
  interface GlobalThis {
    IS_REACT_ACT_ENVIRONMENT?: boolean
  }
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true
```

### BudgetStatus Type Mismatch

**Problem**: `status: string` not assignable to union type

**Solution**:
```typescript
const mockBudgetStatus: BudgetStatus = {
  status: "ok" as const,  // Use 'as const' for literal type
  // ... other fields
}
```

### Immer State Type Errors

**Problem**: `WritableDraft<T>` not assignable to `T`

**Solution**: Don't store DOM elements in Immer state. Use refs or selectors instead:
```typescript
// Instead of:
state.imageRef = someImageElement

// Use:
const imageRef = useRef<HTMLImageElement>(null)
// Or use state for data, refs for DOM
```

## Verification

After fixes, run:
```bash
cd web_app
npm run build  # Runs tsc && vite build
```

## See Also

- [Code Style](code-style.md)
- [Terminal Statuses](terminal-statuses.md)
- [Development Guide](../guides/development.md)
