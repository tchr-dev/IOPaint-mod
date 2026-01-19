# Settings Component Test Implementation Plan

## Overview

This document outlines the implementation plan for testing the Settings.tsx
component, which is one of the most complex components in the IOPaint frontend.

## Component Analysis

The Settings component is a complex form with multiple sections:

1. Model selection
2. Manual inpainting toggle
3. Download mask toggle
4. Auto extract prompt toggle
5. OpenAI provider selection
6. Image tool mode selection
7. Budget limit inputs (daily, monthly, session)

## Test Requirements

### 1. Context Provider Setup

- Mock next-themes for theme handling
- Mock TooltipProvider for tooltip components
- Mock useStore for state management

### 2. State Management

- Mock serverConfig response
- Set up initial state for settings
- Set up initial state for OpenAI settings
- Set up initial state for file manager

### 3. Test Cases

#### Form Rendering Tests

- Should render settings dialog when triggered
- Should display all form fields
- Should show correct initial values
- Should render tabs correctly

#### Model Selection Tests

- Should display available models in dropdown
- Should allow model selection
- Should show mapped preset for selected model
- Should handle single model case (disabled dropdown)

#### Toggle Tests

- Should toggle manual inpainting setting
- Should toggle download mask setting
- Should toggle auto extract prompt setting

#### Provider Selection Tests

- Should display all OpenAI providers
- Should allow provider selection
- Should show correct default value

#### Tool Mode Selection Tests

- Should display all tool modes
- Should allow tool mode selection
- Should show correct default value

#### Budget Limit Tests

- Should display budget limit inputs
- Should allow budget limit editing
- Should handle zero values (unlimited)
- Should handle decimal values
- Should update budget limits on submit

#### Form Submission Tests

- Should save settings on form submit
- Should switch models when different model selected
- Should fetch OpenAI capabilities when provider changes
- Should update budget limits when values change

### 4. Edge Cases

- Empty model list handling
- Single model handling (disabled dropdown)
- Invalid budget limit values
- Network errors during model switch
- Network errors during budget limit update

## Implementation Strategy

### Test File Structure

```
Settings.test.tsx
├── Imports and mocks
├── Helper functions
├── beforeEach setup
├── Rendering tests
├── Model selection tests
├── Toggle tests
├── Provider selection tests
├── Tool mode selection tests
├── Budget limit tests
├── Form submission tests
└── Edge case tests
```

### Mock Setup

```tsx
// Mock next-themes
vi.mock("next-themes", () => ({
    useTheme: () => ({
        theme: "light",
        setTheme: vi.fn(),
    }),
}));

// Mock Tooltip components
vi.mock("@/components/ui/tooltip", () => ({
    Tooltip: ({ children }: any) => children,
    TooltipTrigger: ({ children }: any) => children,
    TooltipContent: ({ children }: any) => children,
}));

// Mock useStore
vi.mock("@/lib/states", () => ({
    useStore: vi.fn(),
}));

// Mock API functions
vi.mock("@/lib/api", () => ({
    getServerConfig: vi.fn(),
    switchModel: vi.fn(),
}));

// Mock react-query
vi.mock("@tanstack/react-query", () => ({
    useQuery: vi.fn(),
}));
```

### State Setup

```tsx
const mockServerConfig = {
    modelInfos: [
        { name: "lama", model_type: "inpaint" },
        { name: "sd", model_type: "diffusers_sd" },
    ],
    // ... other config
};

const mockSettings = {
    model: { name: "lama", model_type: "inpaint" },
    enableManualInpainting: true,
    enableDownloadMask: false,
    enableAutoExtractPrompt: true,
    openAIProvider: "server",
    openAIToolMode: "local",
    // ... other settings
};

const resetState = () => {
    (useStore as any).mockImplementation(() => [
        vi.fn(), // updateAppState
        mockSettings,
        vi.fn(), // updateSettings
        { inputDirectory: "", outputDirectory: "" }, // fileManagerState
        vi.fn(), // setAppModel
        vi.fn(), // setServerConfig
        vi.fn(), // fetchOpenAICapabilities
        false, // isOpenAIMode
        null, // budgetLimits
        vi.fn(), // refreshBudgetLimits
        vi.fn(), // updateBudgetLimits
    ]);

    vi.mocked(useQuery).mockReturnValue({
        data: mockServerConfig,
        status: "success",
        refetch: vi.fn(),
    } as any);
};
```

### Test Implementation

#### 1. Basic Rendering Test

```tsx
it("renders settings dialog with all form fields", async () => {
    // Setup
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    // Render
    await act(async () => {
        root.render(<SettingsDialog />);
    });

    // Assertions
    expect(container.querySelector("dialog")).toBeTruthy();
    expect(container.querySelector("select[name='model']")).toBeTruthy();
    expect(container.querySelector("input[name='enableManualInpainting']"))
        .toBeTruthy();
    // ... other fields

    // Cleanup
    await act(async () => {
        root.unmount();
    });
    container.remove();
});
```

#### 2. Model Selection Test

```tsx
it("allows model selection and shows mapped preset", async () => {
    // Setup
    const updateSettings = vi.fn();
    const setAppModel = vi.fn();
    const switchModel = vi.fn().mockResolvedValue({ name: "sd" })(
        useStore as any,
    ).mockImplementation(() => [
        vi.fn(), // updateAppState
        mockSettings,
        updateSettings,
        { inputDirectory: "", outputDirectory: "" }, // fileManagerState
        setAppModel,
        vi.fn(), // setServerConfig
        vi.fn(), // fetchOpenAICapabilities
        false, // isOpenAIMode
        null, // budgetLimits
        vi.fn(), // refreshBudgetLimits
        vi.fn(), // updateBudgetLimits
    ]);

    vi.mocked(switchModel).mockResolvedValue({ name: "sd" });

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    await act(async () => {
        root.render(<SettingsDialog />);
    });

    // Simulate model selection
    const select = container.querySelector(
        "select[name='model']",
    ) as HTMLSelectElement;
    expect(select).toBeTruthy();

    if (select) {
        await act(async () => {
            select.value = "sd";
            select.dispatchEvent(new Event("change", { bubbles: true }));
        });

        // Submit form
        const form = container.querySelector("form");
        if (form) {
            await act(async () => {
                form.dispatchEvent(new Event("submit", { bubbles: true }));
            });
        }
    }

    // Assertions
    expect(updateSettings).toHaveBeenCalled();
    expect(setAppModel).toHaveBeenCalled();

    await act(async () => {
        root.unmount();
    });
    container.remove();
});
```

#### 3. Budget Limit Test

```tsx
it("updates budget limits when values change", async () => {
    // Setup with budget limits
    const updateBudgetLimits = vi.fn();
    const mockBudgetLimits = {
        dailyCapUsd: 10,
        monthlyCapUsd: 100,
        sessionCapUsd: 5,
    }(useStore as any).mockImplementation(() => [
        vi.fn(), // updateAppState
        mockSettings,
        vi.fn(), // updateSettings
        { inputDirectory: "", outputDirectory: "" }, // fileManagerState
        vi.fn(), // setAppModel
        vi.fn(), // setServerConfig
        vi.fn(), // fetchOpenAICapabilities
        true, // isOpenAIMode
        mockBudgetLimits,
        vi.fn(), // refreshBudgetLimits
        updateBudgetLimits, // updateBudgetLimits
    ]);

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    await act(async () => {
        root.render(<SettingsDialog />);
    });

    // Change budget limit values
    const dailyInput = container.querySelector(
        "input[name='openAIDailyCapUsd']",
    ) as HTMLInputElement;
    if (dailyInput) {
        await act(async () => {
            dailyInput.value = "20";
            dailyInput.dispatchEvent(new Event("input", { bubbles: true }));
        });
    }

    // Submit form
    const form = container.querySelector("form");
    if (form) {
        await act(async () => {
            form.dispatchEvent(new Event("submit", { bubbles: true }));
        });
    }

    // Assertions
    expect(updateBudgetLimits).toHaveBeenCalledWith({
        dailyCapUsd: 20,
        monthlyCapUsd: 100,
        sessionCapUsd: 5,
    });

    await act(async () => {
        root.unmount();
    });
    container.remove();
});
```

## Test Data

### Mock Models

```tsx
const mockModelInfos = [
    { name: "lama", model_type: "inpaint", support_controlnet: false },
    { name: "sd", model_type: "diffusers_sd", support_controlnet: true },
    { name: "anytext", model_type: "anytext", support_controlnet: false },
];
```

### Mock Server Config

```tsx
const mockServerConfig = {
    modelInfos: mockModelInfos,
    plugins: [],
    removeBGModel: "briaai/RMBG-1.4",
    removeBGModels: [],
    realesrganModel: "realesr-general-x4v3",
    realesrganModels: [],
    interactiveSegModel: "vit_b",
    interactiveSegModels: [],
    enableFileManager: false,
    enableAutoSaving: false,
    enableControlnet: false,
    controlnetMethod: "lllyasviel/control_v11p_sd15_canny",
    disableModelSwitch: false,
    isDesktop: false,
    samplers: ["DPM++ 2M SDE Karras"],
};
```

### Mock Settings

```tsx
const mockSettings = {
    model: mockModelInfos[0],
    enableDownloadMask: false,
    enableManualInpainting: true,
    enableUploadMask: false,
    enableAutoExtractPrompt: true,
    openAIProvider: "server",
    openAIToolMode: "local",
    showCropper: false,
    showExtender: false,
    extenderDirection: "xy",
    ldmSteps: 30,
    ldmSampler: "ddim",
    zitsWireframe: true,
    cv2Radius: 5,
    cv2Flag: "INPAINT_NS",
    prompt: "",
    negativePrompt:
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    seed: 42,
    seedFixed: false,
    sdMaskBlur: 12,
    sdStrength: 1.0,
    sdSteps: 50,
    sdGuidanceScale: 7.5,
    sdSampler: "DPM++ 2M",
    sdMatchHistograms: false,
    sdScale: 1.0,
    p2pImageGuidanceScale: 1.5,
    enableControlnet: false,
    controlnetMethod: "lllyasviel/control_v11p_sd15_canny",
    controlnetConditioningScale: 0.4,
    enableBrushNet: false,
    brushnetMethod: "random_mask",
    brushnetConditioningScale: 1.0,
    enableLCMLora: false,
    enablePowerPaintV2: false,
    powerpaintTask: "text_guided",
    adjustMaskKernelSize: 12,
};
```

## Edge Cases to Test

1. **Empty Model List**
   - Should handle gracefully when no models are available
   - Should show appropriate message

2. **Single Model**
   - Should disable model selection dropdown
   - Should show the single model

3. **Invalid Budget Values**
   - Should handle negative values
   - Should handle non-numeric values
   - Should handle very large values

4. **Network Errors**
   - Should show error message when model switch fails
   - Should show error message when budget update fails
   - Should maintain form state on error

5. **Form Validation**
   - Should validate required fields
   - Should show validation errors
   - Should prevent submission with invalid data

## Test Execution Plan

1. **Phase 1: Basic Rendering**
   - Test dialog rendering
   - Test form field presence
   - Test initial values

2. **Phase 2: Simple Interactions**
   - Test toggle switches
   - Test dropdown selections
   - Test input changes

3. **Phase 3: Complex Interactions**
   - Test model switching
   - Test budget limit updates
   - Test form submission

4. **Phase 4: Edge Cases**
   - Test error handling
   - Test validation
   - Test boundary conditions

## Expected Outcomes

1. **Coverage Goals**
   - 100% of Settings component functionality covered
   - All user interactions tested
   - All state transitions tested
   - All error conditions handled

2. **Quality Metrics**
   - All tests pass
   - No console errors
   - No memory leaks
   - Fast execution (< 5 seconds)

3. **Maintenance**
   - Clear test structure
   - Descriptive test names
   - Comprehensive assertions
   - Minimal mocking where possible
