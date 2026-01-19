import React, { type ReactNode } from "react"
import { act, render, type RenderOptions, type RenderResult } from "@testing-library/react"
import { useStore } from "@/lib/states"
import { TooltipProvider } from "@/components/ui/tooltip"

interface WrapperProps {
  children: ReactNode
}

const defaultStoreState = {
  file: null,
  paintByExampleFile: null,
  customMask: null,
  imageHeight: 512,
  imageWidth: 512,
  isInpainting: false,
  rendersCountBeforeInpaint: 0,
  isAdjustingMask: false,
  disableShortCuts: false,
  editorState: {
    baseBrushSize: 12,
    brushSizeScale: 1,
    renders: [],
    extraMasks: [],
    prevExtraMasks: [],
    temporaryMasks: [],
    lineGroups: [],
    lastLineGroup: [],
    curLineGroup: [],
    redoRenders: [],
    redoCurLines: [],
    redoLineGroups: [],
  },
  interactiveSegState: {
    isInteractiveSeg: false,
    tmpInteractiveSegMask: null,
    clicks: [],
  },
  cropperState: { x: 0, y: 0, width: 512, height: 512 },
  extenderState: { x: 0, y: 0, width: 512, height: 512 },
  isCropperExtenderResizing: false,
  settings: {
    enableManualInpainting: true,
    showCropper: false,
    showExtender: false,
    enableBrushNet: false,
    enableControlnet: false,
    enableLCMLora: false,
    enablePowerPaintV2: true,
    brushnetConditioningScale: 1.0,
    brushnetMethod: "brushnet_alpha",
    controlnetConditioningScale: 1.0,
    controlnetMethod: "control_v11p_sd15_inpaint",
    prompt: "",
    negativePrompt: "",
    powerpaintTask: "text_guided",
    sdSteps: 30,
    sdGuidanceScale: 7.5,
    sdStrength: 0.75,
    p2pImageGuidanceScale: 1.5,
    sdMaskBlur: 12,
    sdMatchHistograms: false,
    adjustMaskKernelSize: 12,
    seedFixed: false,
    seed: 42,
    sdSampler: "Euler",
    model: {
      name: "lama",
      support_brushnet: false,
      support_controlnet: false,
      support_lcm_lora: false,
      support_powerpaint_v2: true,
      support_outpainting: false,
      support_strength: true,
      need_prompt: true,
      brushnets: {},
      controlnets: {},
    },
    sampler: ["Euler", "Euler a", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras", "DPM++ 2M a", "DPM fast"],
  },
  serverConfig: {
    samplers: ["Euler", "Euler a", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras", "DPM++ 2M a", "DPM fast"],
    models: [],
  },
}

export function resetStoreState() {
  useStore.setState({
    ...defaultStoreState,
    settings: {
      ...defaultStoreState.settings,
      ...useStore.getState().settings,
    },
    editorState: {
      ...defaultStoreState.editorState,
      ...useStore.getState().editorState,
    },
    interactiveSegState: {
      ...defaultStoreState.interactiveSegState,
      ...useStore.getState().interactiveSegState,
    },
    cropperState: {
      ...defaultStoreState.cropperState,
      ...useStore.getState().cropperState,
    },
    extenderState: {
      ...defaultStoreState.extenderState,
      ...useStore.getState().extenderState,
    },
  })
}

function AllTheProviders({ children }: WrapperProps) {
  return (
    <TooltipProvider>
      {children}
    </TooltipProvider>
  )
}

export function renderWithProviders(
  ui: React.ReactElement,
  options?: RenderOptions
): RenderResult {
  return render(ui, {
    wrapper: AllTheProviders,
    ...options,
  })
}

export async function actAsync(fn: () => Promise<void> | void): Promise<void> {
  await act(async () => {
    await fn()
  })
}

export function createMockRef<T>(initialValue: T | null = null): React.MutableRefObject<T | null> {
  return {
    current: initialValue,
  }
}

export function mockEvent(type: string, options: EventInit = {}): Event {
  return new Event(type, { bubbles: true, ...options })
}

export function mockMouseEvent(
  type: string,
  options: MouseEventInit = {}
): MouseEvent {
  return new MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    view: window,
    ...options,
  })
}

export function mockKeyboardEvent(
  type: string,
  options: KeyboardEventInit = {}
): KeyboardEvent {
  return new KeyboardEvent(type, {
    bubbles: true,
    cancelable: true,
    view: window,
    ...options,
  })
}

export function expectElementToHaveText(
  container: HTMLElement,
  selector: string,
  expectedText: string
): void {
  const element = container.querySelector(selector)
  expect(element).toBeTruthy()
  expect(element?.textContent).toContain(expectedText)
}

export function expectElementToBeVisible(
  container: HTMLElement,
  selector: string
): void {
  const element = container.querySelector(selector)
  expect(element).toBeTruthy()
  expect(element).toBeVisible()
}

export function expectElementToBeHidden(
  container: HTMLElement,
  selector: string
): void {
  const element = container.querySelector(selector)
  if (element) {
    expect(element).not.toBeVisible()
  } else {
    expect(true).toBe(true)
  }
}

export function getStoreState() {
  return {
    ...defaultStoreState,
    settings: useStore.getState().settings,
    editorState: useStore.getState().editorState,
    interactiveSegState: useStore.getState().interactiveSegState,
    cropperState: useStore.getState().cropperState,
    extenderState: useStore.getState().extenderState,
  }
}
