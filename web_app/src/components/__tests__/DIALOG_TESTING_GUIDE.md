# Dialog Testing Guide

## Problem: Radix UI Dialog Components in Tests

Radix UI Dialog components render their content in **portals** (outside the React component tree), which causes issues when testing:

1. Dialog content appears in `document.body`, not in the test `container`
2. Clicking buttons doesn't automatically open dialogs due to state management
3. TooltipProvider context is required for buttons with tooltips

## Solution Pattern

### 1. Add Required Mocks

Always include these mocks at the top of dialog component test files:

```typescript
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"

// Mock Tooltip components (required for IconButton with tooltips)
vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => children,
  TooltipTrigger: ({ children }: any) => children,
  TooltipContent: ({ children }: any) => children,
  TooltipProvider: ({ children }: any) => children,
}))

describe("YourComponent", () => {
  beforeEach(() => {
    // Use conditional check for localStorage
    if (typeof localStorage !== "undefined" && localStorage.clear) {
      localStorage.clear()
    }
  })
})
```

### 2. Check `document.body` Instead of `container`

Since dialog content renders in a portal, always check `document.body.textContent`:

```typescript
// ❌ WRONG - Dialog content is NOT in container
expect(container.textContent).toContain("Dialog Title")

// ✅ CORRECT - Dialog content is in document.body
expect(document.body.textContent).toContain("Dialog Title")
```

### 3. Handle Controlled Dialogs

For dialogs that use state management (like `useToggle`), mock the state to be open:

```typescript
// For components using useToggle from @uidotdev/usehooks
vi.mock("@uidotdev/usehooks", () => ({
  useToggle: () => [true, vi.fn()], // Return true to keep dialog open
}))
```

### 4. Mock React Query for Complex Components

If a dialog component uses `useQuery`, mock it:

```typescript
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({
    data: mockData,
    status: "success",
    refetch: vi.fn(),
  }),
}))
```

## Complete Example: Testing a Dialog Component

```typescript
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act } from "react-dom/test-utils"
import { createRoot } from "react-dom/client"
import { MyDialogComponent } from "../MyDialogComponent"

// 1. Mock Tooltips
vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => children,
  TooltipTrigger: ({ children }: any) => children,
  TooltipContent: ({ children }: any) => children,
  TooltipProvider: ({ children }: any) => children,
}))

// 2. Mock controlled dialog state (if needed)
vi.mock("@uidotdev/usehooks", () => ({
  useToggle: () => [true, vi.fn()],
}))

describe("MyDialogComponent", () => {
  beforeEach(() => {
    if (typeof localStorage !== "undefined" && localStorage.clear) {
      localStorage.clear()
    }
  })

  it("renders dialog trigger button", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<MyDialogComponent />)
    })

    // Trigger button is in container
    expect(container.querySelector("button")).toBeTruthy()

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it("shows dialog content", async () => {
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<MyDialogComponent />)
    })

    const button = container.querySelector("button")
    if (button) {
      await act(async () => {
        button.click()
      })
    }

    // ✅ Check document.body for portal content
    expect(document.body.textContent).toContain("Dialog Title")
    expect(document.body.textContent).toContain("Dialog Content")

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})
```

## Common Patterns

### Uncontrolled Dialog (Coffee component)
```typescript
// Component uses <Dialog> without open prop
// No special mocking needed, just check document.body
expect(document.body.textContent).toContain("Dialog Content")
```

### Controlled Dialog with useToggle (Shortcuts, Settings)
```typescript
// Component uses useToggle(false) for open state
vi.mock("@uidotdev/usehooks", () => ({
  useToggle: () => [true, vi.fn()], // Mock as open
}))
```

### Dialog with useQuery (Settings)
```typescript
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({
    data: mockServerConfig,
    status: "success",
    refetch: vi.fn(),
  }),
}))
```

## Checklist for Dialog Tests

- [ ] Add Tooltip mocks
- [ ] Use `document.body.textContent` for dialog content assertions
- [ ] Mock `useToggle` to return `[true, vi.fn()]` if dialog uses it
- [ ] Mock `useQuery` if dialog fetches data
- [ ] Use conditional localStorage.clear()
- [ ] Clean up with `root.unmount()` and `container.remove()`
- [ ] Wrap interactions in `act(async () => {...})`

## Why This Works

1. **Portals**: Radix UI Dialog uses React portals, rendering content at `document.body` level
2. **State Management**: Mocking controlled state as "open" bypasses click/interaction complexity
3. **Context Providers**: Mocking Tooltip components avoids TooltipProvider requirement
4. **Act Boundaries**: Wrapping in `act()` ensures React state updates complete

## Files Using This Pattern

- ✅ Coffee.test.tsx
- ✅ Shortcuts.test.tsx
- 🔄 Settings.test.tsx (pending)
- 🔄 FileManager.test.tsx (pending)
- 🔄 GenerationHistory.test.tsx (pending)
