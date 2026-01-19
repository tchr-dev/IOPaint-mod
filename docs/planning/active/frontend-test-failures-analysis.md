# Frontend Test Failures Analysis

**Date**: 2026-01-19  
**Status**: In Progress  
**Related Todo**: Fix existing failing tests

## Summary

**Total Test Files**: 16 (9 failed, 7 passed)  
**Total Tests**: 153 (19 failed, 134 passed)  
**Success Rate**: 87.6%

## Test Failures by File

### 1. Settings.test.tsx (6 failures)
**Component**: Settings dialog component  
**Failures**:
- renders settings dialog with all form fields
- displays correct initial values  
- allows model selection and shows mapped preset
- updates budget limits when values change
- handles single model case (disabled dropdown)
- shows correct mapped preset for selected model

**Root Cause**: Dialog not opening when button is clicked. The component uses Radix UI Dialog which renders in a portal. The `useToggle` hook state is not being updated properly in tests, causing the dialog to remain closed.

**Symptoms**:
- Tests click the settings button but dialog doesn't open
- `document.querySelector("[role='dialog']")` returns `null`
- Form fields cannot be found because dialog content is not in DOM

**Fix Strategy**:
1. Mock the `useToggle` hook to control dialog open state
2. Or render the Settings component with `open=true` prop if available
3. Or test the component without dialog wrapper (test content directly)

---

### 2. Cropper.test.tsx (2 failures)
**Component**: Cropper resizable overlay component  
**Failures**:
- renders cropper when show is true
- does not render when show is false

**Root Cause**: `TypeError: setY is not a function` at Cropper.tsx:96

**Symptoms**:
```
TypeError: setY is not a function
    at /Users/ikovale/Dev/IOPaint-mod/web_app/src/components/Cropper.tsx:96:5
```

**Analysis**: The component expects certain functions from `useStore` but the mock is not providing them correctly. The mock needs to match the exact structure of what `useStore` returns for Cropper.

**Fix Strategy**:
1. Read Cropper.tsx lines 40-100 to see what state it expects
2. Update the mock to provide all required functions including `setY`
3. Ensure mock return value matches the array destructuring pattern

---

### 3. OpenAIEditPanel.test.tsx (1 failure)
**Component**: OpenAI edit panel for image editing  
**Failures**:
- renders edit panel UI

**Root Cause**: `TypeError: Cannot read properties of undefined (reading 'images_edit')` at OpenAIEditPanel.tsx:86

**Symptoms**:
```
TypeError: Cannot read properties of undefined (reading 'images_edit')
    at OpenAIEditPanel (/Users/ikovale/Dev/IOPaint-mod/web_app/src/components/OpenAI/OpenAIEditPanel.tsx:86:36)
```

**Analysis**: The component is trying to access a capabilities object that isn't mocked. Line 86 is likely checking `capabilities.images_edit` or similar.

**Fix Strategy**:
1. Read OpenAIEditPanel.tsx line 86 to see what object is being accessed
2. Add proper capabilities mock to the useStore mock
3. Ensure the capabilities object has the required structure

---

### 4. Coffee.test.tsx (2 failures)
**Component**: Coffee donation button/dialog  
**Failures**:
- renders coffee icon button
- opens dialog when coffee button is clicked

**Root Cause**: Similar to Settings - dialog-based component not rendering properly

**Fix Strategy**:
1. Check if Coffee component uses Radix UI Dialog
2. Apply same fix strategy as Settings component
3. May need to mock dialog open state

---

### 5. FileManager.test.tsx (1 failure)
**Component**: File manager dialog  
**Failures**:
- renders file manager dialog trigger button

**Root Cause**: Likely useStore mock incomplete or API mock not set up correctly

**Fix Strategy**:
1. Check component requirements
2. Update mocks to match expected structure
3. Add any missing API mocks

---

### 6. Shortcuts.test.tsx (2 failures)
**Component**: Keyboard shortcuts dialog  
**Failures**:
- renders shortcuts icon button
- opens dialog when shortcuts button is clicked

**Root Cause**: Similar to Coffee/Settings - dialog component issue

**Fix Strategy**:
- Apply same dialog fix strategy as Settings

---

### 7. Header.test.tsx (3 failures)
**Component**: Application header  
**Failures**:
- renders header with upload button
- renders local/cloud mode toggle
- renders settings and theme buttons

**Root Cause**: Likely useStore mock or component requirements not met

**Fix Strategy**:
1. Review Header component dependencies
2. Update useStore mock
3. Add any missing component mocks

---

### 8. GenerationHistory.test.tsx (1 failure)
**Component**: OpenAI generation history panel  
**Failures**:
- renders history panel

**Root Cause**: Likely useStore mock incomplete

**Fix Strategy**:
- Check component dependencies and update mocks

---

### 9. capabilities-ui.test.tsx (1 failure)
**Component**: OpenAI capabilities UI component  
**Failures**:
- renders size options from capabilities

**Root Cause**: Capabilities data not mocked correctly

**Fix Strategy**:
- Add proper capabilities mock with size options

---

## Common Patterns

### Pattern 1: Dialog Components Not Opening
**Files Affected**: Settings, Coffee, Shortcuts  
**Issue**: Radix UI Dialog components render in portals and require proper state management

### Pattern 2: Incomplete useStore Mocks
**Files Affected**: Cropper, FileManager, Header, GenerationHistory, OpenAIEditPanel  
**Issue**: Mock return values don't match component expectations

### Pattern 3: Missing Capabilities/API Mocks
**Files Affected**: OpenAIEditPanel, capabilities-ui  
**Issue**: OpenAI-related components need capabilities object mocked

---

## Action Items

1. **High Priority** - Fix Settings.test.tsx (6 tests)
   - Investigate dialog state management
   - Find way to test dialog content without interaction

2. **High Priority** - Fix Cropper.test.tsx (2 tests)
   - Read Cropper component to understand state requirements
   - Update useStore mock with correct structure

3. **High Priority** - Fix OpenAI component tests (3 tests)
   - Add capabilities mock
   - Ensure all OpenAI state is properly mocked

4. **Medium Priority** - Fix Coffee/Shortcuts tests (4 tests)
   - Apply dialog fix strategy

5. **Medium Priority** - Fix Header/FileManager/History tests (5 tests)
   - Review and update useStore mocks

---

## Testing Infrastructure Improvements Needed

1. **Create a test utils file** with:
   - Common mock factories (useStore, capabilities, API responses)
   - Dialog testing utilities
   - Reusable test setup functions

2. **Document mock patterns** for:
   - Zustand store mocking
   - Radix UI component testing
   - Portal component testing

3. **Add test helpers** for:
   - Opening dialogs programmatically
   - Waiting for portal content
   - Finding elements in portals

---

## Next Steps

1. Read failing component source code to understand requirements
2. Create comprehensive useStore mock factories
3. Fix tests one file at a time, starting with highest priority
4. Extract common patterns into reusable utilities
5. Document solutions for future test development
