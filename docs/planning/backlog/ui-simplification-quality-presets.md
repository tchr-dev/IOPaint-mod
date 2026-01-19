# IOPaint UI Simplification Plan

> **Status:** Ready for implementation
> **Created:** 2026-01-18
> **Scope:** Frontend UI overhaul with Quality Preset system

## Goal

Remove complex settings and provide a clear, self-explanatory UI with minimal toolbar items. Focus on user outcomes (Fast/Cheap/Perfect) rather than technical model names.

---

## Target UI Layout

### Header Toolbar (5 items - down from 10)

| Keep | Item | Purpose |
|------|------|---------|
| ✓ | Upload Image | Load image to edit |
| ✓ | Local/Cloud Toggle | Switch between local AI and cloud API |
| ✓ | Shortcuts | Show keyboard shortcuts help |
| ✓ | Settings | Open minimal settings dialog |
| ✓ | Theme Toggle | Light/Dark/System |

**Remove:** File Manager, Upload Mask, Play Mask, Rerun Mask, Prompt Input (moved to side panel)

### Bottom Toolbar (6 items)

| Item | Purpose |
|------|---------|
| Brush Size Slider | Adjust drawing brush |
| Undo | Undo last action |
| Redo | Redo undone action |
| **Remove Background** | NEW - Direct plugin action |
| **Upscale 2x** | NEW - Direct plugin action |
| Save | Export result |

**Remove:** Reset Zoom (use Esc key), Show Original (use Tab key), Manual Inpaint (use Shift+R)

### Side Panel (Simplified)

- **Quality Preset Selector**: Fast / Balanced / Best Quality
- **Prompt Input**: Only shown when using diffusion models (Best Quality)
- **Helper text**: Explains current selection

### Settings Dialog (2 tabs - down from 3)

- **General Tab**: Model dropdown (flat list), essential toggles, budget limits
- **Plugins Tab**: Plugin model selection (unchanged)

**Remove:** Model Tab with complex categorization

---

## Quality Preset System

| Preset | Maps To | Description |
|--------|---------|-------------|
| Fast | `lama` | Quick inpainting for previews |
| Balanced | `mat` | Good quality, reasonable speed |
| Best Quality | SD/SDXL (auto-select) | Highest quality, requires prompt |

### Smart Model Resolution (for "Best Quality")

Priority order when selecting SD model:
1. SD 1.5 Inpaint (optimized for inpainting)
2. SD 1.5 (general purpose)
3. SDXL Inpaint
4. SDXL
5. Fallback to MAT if no SD available

---

## Files to Create

### 1. `web_app/src/lib/presets.ts` (~100 lines)

```typescript
export enum QualityPreset {
  FAST = "fast",
  BALANCED = "balanced",
  BEST = "best",
  CUSTOM = "custom"
}

export interface PresetConfig {
  modelName: string
  displayName: string
  description: string
}

export const PRESET_CONFIGS: Record<QualityPreset, PresetConfig> = {
  [QualityPreset.FAST]: {
    modelName: "lama",
    displayName: "Fast",
    description: "Quick inpainting for previews"
  },
  [QualityPreset.BALANCED]: {
    modelName: "mat",
    displayName: "Balanced",
    description: "Good quality with reasonable speed"
  },
  [QualityPreset.BEST]: {
    modelName: "auto-sd",
    displayName: "Best Quality",
    description: "Highest quality using Stable Diffusion (requires prompt)"
  },
  [QualityPreset.CUSTOM]: {
    modelName: "",
    displayName: "Custom",
    description: "Manually selected model"
  }
}

export function resolveModelForPreset(
  preset: QualityPreset,
  availableModels: ModelInfo[]
): ModelInfo | null { /* ... */ }

export function detectPresetFromModel(model: ModelInfo): QualityPreset { /* ... */ }
```

### 2. `web_app/src/components/SidePanel/SimplifiedOptions.tsx` (~150 lines)

- Quality preset Select dropdown
- Conditional prompt Textarea (only when model.need_prompt)
- Toast notifications for model switching
- Error handling when model unavailable

---

## Files to Modify

### 3. `web_app/src/lib/states.ts`

**Changes:**
- Add `qualityPreset: QualityPreset` to Settings interface
- Add default: `qualityPreset: QualityPreset.FAST`
- Bump persist version: `version: 5`
- Add migration logic for backward compatibility

### 4. `web_app/src/components/Header.tsx` (-90 lines)

**Remove:**
- Lines 101-105: File Manager button
- Lines 117-183: Upload Mask + Play Mask sections
- Lines 185-197: Rerun Mask button
- Lines 230-233: Centered Prompt Input

**Keep:**
- Upload Image button
- Local/Cloud toggle (CRITICAL for OpenAI mode)
- Shortcuts, Settings, Theme buttons

### 5. `web_app/src/components/Editor.tsx` (bottom toolbar)

**Remove from toolbar (lines 1012-1096):**
- Reset Zoom button (lines 1025-1031)
- Show Original button (lines 1046-1070)
- Manual Inpaint button (lines 1079-1094)

**Add to toolbar:**
```typescript
<IconButton
  tooltip="Remove Background"
  disabled={!renders.length || isProcessing || !removeBGEnabled}
  onClick={() => runRenderablePlugin(false, "RemoveBG")}
>
  <Slice className="h-4 w-4" />
</IconButton>

<IconButton
  tooltip="Upscale 2x"
  disabled={!renders.length || isProcessing || !upscaleEnabled}
  onClick={() => runRenderablePlugin(false, "RealESRGAN", { upscale: 2 })}
>
  <Fullscreen className="h-4 w-4" />
</IconButton>
```

### 6. `web_app/src/components/SidePanel/index.tsx`

**Change routing (lines 41-55):**
```typescript
const renderSidePanelOptions = () => {
  if (isOpenAIMode) {
    return isOpenAIEditMode ? <OpenAIEditPanel /> : <OpenAIGeneratePanel />
  }
  // Simplified local mode - single panel for all models
  return <SimplifiedOptions />
}
```

**Remove imports:** LDMOptions, DiffusionOptions, CV2Options

### 7. `web_app/src/components/Settings.tsx` (-180 lines)

**Remove:**
- TAB_MODEL constant and tab
- renderModelSettings() function (lines 574-645)
- renderOpenAISettings() function (lines 408-572)

**Add to renderGeneralSettings():**
- Flat model dropdown with all available models
- Import and use detectPresetFromModel() to sync preset when model changes

---

## Implementation Phases

### Phase 1: Core Infrastructure
1. Create `web_app/src/lib/presets.ts`
2. Update `web_app/src/lib/states.ts` with qualityPreset state
3. Test: Clear localStorage, reload, verify no errors

### Phase 2: SimplifiedOptions Component
1. Create `web_app/src/components/SidePanel/SimplifiedOptions.tsx`
2. Update `web_app/src/components/SidePanel/index.tsx` routing
3. Test: Open side panel, change presets, verify model switches

### Phase 3: Header Simplification
1. Remove buttons from `web_app/src/components/Header.tsx`
2. Clean up unused imports and state subscriptions
3. Test: Header shows 5 buttons, Local/Cloud toggle works

### Phase 4: Bottom Toolbar Update
1. Modify `web_app/src/components/Editor.tsx`
2. Add plugin action buttons
3. Test: RemoveBG and Upscale buttons work

### Phase 5: Settings Dialog
1. Simplify `web_app/src/components/Settings.tsx`
2. Test: 2 tabs, model dropdown works

---

## Verification Plan

### Functional Tests
1. **Preset switching**: Fast → Balanced → Best, verify model switches
2. **Plugin buttons**: RemoveBG and Upscale work correctly
3. **Local/Cloud toggle**: Both modes function properly
4. **Backward compatibility**: Old localStorage migrates correctly
5. **Missing models**: Error handling shows toast

### Keyboard Shortcuts (must still work)
- `Esc` - Reset zoom
- `Tab` - Show original image
- `Ctrl+Z` - Undo
- `Shift+Ctrl+Z` - Redo
- `Shift+R` - Manual inpaint
- `S` - Open settings
- `C` - Toggle side panel
- `[` / `]` - Adjust brush size

### Build Verification
```bash
cd web_app
npm run lint  # No errors
npm run build  # Successful build
```

---

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Header buttons | 10 | 5 | -50% |
| Bottom toolbar buttons | 7 | 6 | -14% |
| Side panel lines | 948 | 150 | -84% |
| Settings tabs | 3 | 2 | -33% |
| User decisions for basic inpaint | ~15 | 3 | -80% |

---

## Future TODOs (Not in this implementation)

- [ ] Remove ControlNet, BrushNet, PowerPaint code entirely
- [ ] Remove Cropper/Extender functionality
- [ ] Add advanced mode toggle for power users
- [ ] Implement OpenAI "service" tool mode
- [ ] Add preset parameter overrides (steps, guidance for each preset)

---

## Key Design Decisions

1. **Quality Preset Abstraction** - Maps user intent to technical models
2. **Single Simplified Panel** - Replace LDM/CV2/Diffusion panels with one
3. **Plugin Action Buttons** - Direct access in bottom toolbar
4. **Preserve Local/Cloud Toggle** - Critical for OpenAI functionality
5. **Keyboard Shortcuts Preserved** - All removed buttons have keyboard equivalents
