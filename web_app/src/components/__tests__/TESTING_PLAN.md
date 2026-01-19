# IOPaint Frontend Testing Plan

## Overview

This document outlines the comprehensive testing strategy for the IOPaint
frontend, covering all components that currently lack tests and addressing
issues with existing failing tests.

## Current Test Coverage Status

### Components WITH Tests:

1. Coffee.tsx (Coffee.test.tsx)
2. Editor.tsx (Editor.test.tsx) - 31 tests
3. Header.tsx (Header.test.tsx)
4. Shortcuts.tsx (Shortcuts.test.tsx)
5. OpenAI components:
   - capabilities-ui.test.tsx
   - generation-history.test.tsx
   - lib/**tests**/openai-job-flow.test.ts

### Components WITHOUT Tests:

1. Cropper.tsx
2. DiffusionProgress.tsx
3. Extender.tsx
4. FileManager.tsx
5. FileSelect.tsx
6. ImageSize.tsx
7. InteractiveSeg.tsx
8. PromptInput.tsx
9. Settings.tsx
10. Workspace.tsx
11. Editor sub-components:
    - BrushToolbar.tsx
    - CanvasView.tsx
12. OpenAI sub-components:
    - CostDisplay.tsx
    - CostWarningModal.tsx
    - GenerationHistory.tsx
    - GenerationPresets.tsx
    - HistoryItem.tsx
    - IntentInput.tsx
    - OpenAIEditPanel.tsx
    - OpenAIGeneratePanel.tsx
    - PromptEditor.tsx
13. SidePanel components:
    - CV2Options.tsx
    - DiffusionOptions.tsx
    - LDMOptions.tsx
    - SimplifiedOptions.tsx
14. UI components (All UI components lack individual tests)

## Test Results Summary

- Total test files: 8
- Passing test files: 3
- Failing test files: 5
- Total tests: 46
- Passing tests: 37
- Failing tests: 9

## Issues with Current Tests

Several tests are failing due to:

1. Missing context providers (TooltipProvider)
2. localStorage mock issues
3. Incorrect assertions
4. Missing act() wrappers

## Test Patterns and Templates

### 1. Simple Component Test Pattern

For components with minimal state and no complex interactions:

```tsx
import { beforeEach, describe, expect, it } from "vitest";
import { act } from "react-dom/test-utils";
import { createRoot } from "react-dom/client";

import SimpleComponent from "../SimpleComponent";

describe("SimpleComponent", () => {
    beforeEach(() => {
        localStorage.clear();
    });

    it("renders correctly", async () => {
        const container = document.createElement("div");
        document.body.appendChild(container);
        const root = createRoot(container);

        await act(async () => {
            root.render(<SimpleComponent />);
        });

        expect(container.querySelector("element")).toBeTruthy();

        await act(async () => {
            root.unmount();
        });
        container.remove();
    });
});
```

### 2. Complex Component Test Pattern (with Zustand state)

For components that interact with the global state:

```tsx
import { beforeEach, describe, expect, it } from "vitest";
import { useStore } from "../../lib/states";

const resetState = () => {
    useStore.setState({
        // Reset specific state properties
        property: defaultValue,
    });
};

describe("ComplexComponent", () => {
    beforeEach(() => {
        localStorage.clear();
        resetState();
    });

    it("handles state changes correctly", () => {
        // Test state interactions
        useStore.getState().action();
        expect(useStore.getState().property).toBe(expectedValue);
    });
});
```

### 3. Component with API Calls Test Pattern

For components that make API calls:

```tsx
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react-dom/test-utils";
import { createRoot } from "react-dom/client";
import { mockApiFunction } from "../../lib/api";

vi.mock("../../lib/api", () => ({
    mockApiFunction: vi.fn(),
}));

describe("ComponentWithApi", () => {
    beforeEach(() => {
        localStorage.clear();
        vi.clearAllMocks();
    });

    it("handles API calls correctly", async () => {
        vi.mocked(mockApiFunction).mockResolvedValue(mockData);

        const container = document.createElement("div");
        document.body.appendChild(container);
        const root = createRoot(container);

        await act(async () => {
            root.render(<ComponentWithApi />);
        });

        expect(mockApiFunction).toHaveBeenCalled();

        await act(async () => {
            root.unmount();
        });
        container.remove();
    });
});
```

### 4. OpenAI Component Test Pattern

For OpenAI-related components:

```tsx
import { beforeEach, describe, expect, it } from "vitest";
import { act } from "react-dom/test-utils";
import { createRoot } from "react-dom/client";
import { useStore } from "../../../lib/states";

const resetOpenAIState = () => {
    const state = useStore.getState();
    useStore.setState({
        openAIState: {
            ...state.openAIState,
            // Reset specific OpenAI state properties
        },
    });
};

describe("OpenAIComponent", () => {
    beforeEach(() => {
        localStorage.clear();
        resetOpenAIState();
    });

    it("handles OpenAI state correctly", async () => {
        // Test OpenAI-specific functionality
        useStore.getState().openAIAction();
        expect(useStore.getState().openAIState.property).toBe(expectedValue);
    });
});
```

### 5. UI Component Test Pattern

For reusable UI components:

```tsx
import { beforeEach, describe, expect, it } from "vitest";
import { act } from "react-dom/test-utils";
import { createRoot } from "react-dom/client";

import { Button } from "../ui/button";

describe("Button", () => {
    beforeEach(() => {
        localStorage.clear();
    });

    it("renders with correct text", async () => {
        const container = document.createElement("div");
        document.body.appendChild(container);
        const root = createRoot(container);

        await act(async () => {
            root.render(<Button>Click me</Button>);
        });

        expect(container.textContent).toContain("Click me");

        await act(async () => {
            root.unmount();
        });
        container.remove();
    });

    it("handles click events", async () => {
        const handleClick = vi.fn();
        const container = document.createElement("div");
        document.body.appendChild(container);
        const root = createRoot(container);

        await act(async () => {
            root.render(<Button onClick={handleClick}>Click me</Button>);
        });

        const button = container.querySelector("button");
        expect(button).toBeTruthy();

        if (button) {
            await act(async () => {
                button.click();
            });
        }

        expect(handleClick).toHaveBeenCalled();

        await act(async () => {
            root.unmount();
        });
        container.remove();
    });
});
```

## Test Infrastructure Improvements

### 1. Context Provider Setup

Many tests fail due to missing context providers. Add this to test setup:

```tsx
// Mock context providers
vi.mock("next-themes", () => ({
    useTheme: () => ({
        theme: "light",
        setTheme: vi.fn(),
    }),
}));

// For components using Tooltip
vi.mock("@/components/ui/tooltip", () => ({
    Tooltip: ({ children }: any) => children,
    TooltipTrigger: ({ children }: any) => children,
    TooltipContent: ({ children }: any) => children,
}));
```

### 2. localStorage Mock Improvements

Ensure proper localStorage mocking:

```tsx
beforeEach(() => {
    localStorage.clear();
    // Reset any localStorage values needed for tests
    localStorage.setItem("key", "value");
});
```

### 3. act() Wrapper Usage

Always wrap React state updates in act():

```tsx
await act(async () => {
    root.render(<Component />);
});

// For user interactions
if (button) {
    await act(async () => {
        button.click();
    });
}
```

## Component Testing Priorities

### High Priority (Complex Components)

1. Settings.tsx - Complex form with many interactions
2. Cropper.tsx - Complex drag/resize interactions
3. Extender.tsx - Complex drag/resize interactions
4. Workspace.tsx - Main application component
5. FileManager.tsx - Complex file management
6. OpenAI components - Critical for OpenAI functionality

### Medium Priority (Standard Components)

1. FileSelect.tsx
2. DiffusionProgress.tsx
3. ImageSize.tsx
4. InteractiveSeg.tsx
5. PromptInput.tsx
6. BrushToolbar.tsx
7. CanvasView.tsx
8. OpenAI sub-components
9. SidePanel components

### Low Priority (UI Components)

1. All UI components in src/components/ui/

## Implementation Strategy

### Phase 1: Fix Existing Tests

1. Add missing context providers
2. Fix localStorage mock issues
3. Add missing act() wrappers
4. Correct failing assertions

### Phase 2: Implement New Tests

1. Start with high-priority components
2. Follow established test patterns
3. Ensure 100% coverage for critical functionality
4. Add edge case testing

### Phase 3: UI Component Tests

1. Test all UI components individually
2. Ensure accessibility compliance
3. Test all variant combinations

## Test Coverage Goals

1. 100% of main components tested
2. 100% of OpenAI components tested
3. 100% of SidePanel components tested
4. 80%+ of UI components tested
5. All edge cases covered
6. All user interactions tested
7. All state transitions tested

## Quality Assurance

1. All tests must pass before merging
2. Code coverage must be >90%
3. All new tests must follow established patterns
4. All tests must be deterministic
5. All tests must run in under 30 seconds total

## Maintenance

1. Update tests when component APIs change
2. Add tests for new components immediately
3. Review test coverage monthly
4. Refactor tests when patterns improve
