import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import DiffusionOptions from "../SidePanel/DiffusionOptions"
import { useStore } from "@/lib/states"
import { PowerPaintTask } from "@/lib/types"

const defaultServerConfig = {
  samplers: ["Euler", "Euler a", "DPM++ 2M", "DPM++ 2M Karras"],
  models: [],
}

const defaultSettings = {
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
  powerpaintTask: PowerPaintTask.text_guided,
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
    name: "powerpaint",
    support_brushnet: true,
    support_controlnet: true,
    support_lcm_lora: true,
    support_powerpaint_v2: true,
    support_outpainting: true,
    support_strength: true,
    need_prompt: true,
    brushnets: {
      brushnet_alpha: "brushnet_alpha",
      brushnet_unofficial: "brushnet_unofficial",
    },
    controlnets: {
      control_v11p_sd15_inpaint: "control_v11p_sd15_inpaint",
      control_v11p_sd15_lineart: "control_v11p_sd15_lineart",
    },
  },
}

const resetStoreState = () => {
  useStore.setState({
    serverConfig: defaultServerConfig,
    settings: defaultSettings,
    paintByExampleFile: null,
    isProcessing: false,
    isInpainting: false,
    updateSettings: vi.fn(),
    runInpainting: vi.fn(),
    cancelInpainting: vi.fn(),
    updateAppState: vi.fn(),
    updateExtenderByBuiltIn: vi.fn(),
    updateExtenderDirection: vi.fn(),
    adjustMask: vi.fn(),
    clearMask: vi.fn(),
    updateEnablePowerPaintV2: vi.fn(),
    updateEnableBrushNet: vi.fn(),
    updateEnableControlnet: vi.fn(),
    updateLCMLora: vi.fn(),
  })
}

describe("DiffusionOptions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStoreState()
  })

  describe("rendering", () => {
    it("renders without crashing", () => {
      render(<DiffusionOptions />)
      expect(screen.getByTestId("diffusion-options")).toBeTruthy()
    })

    it("renders Cropper toggle", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/cropper/i)).toBeTruthy()
    })

    it("renders Extender section when model supports outpainting", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/extender/i)).toBeTruthy()
    })

    it("renders Steps slider", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/steps/i)).toBeTruthy()
    })

    it("renders Guidance slider", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/guidance/i)).toBeTruthy()
    })

    it("renders Sampler select", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/sampler/i)).toBeTruthy()
    })

    it("renders Seed section", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/seed/i)).toBeTruthy()
    })

    it("renders Negative prompt textarea when model needs prompt", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/negative prompt/i)).toBeTruthy()
    })
  })

  describe("Cropper toggle", () => {
    it("toggles showCropper when clicked", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ updateSettings })
      render(<DiffusionOptions />)
      const cropperSwitch = screen.getByLabelText(/cropper/i)
      await userEvent.click(cropperSwitch)
      expect(updateSettings).toHaveBeenCalledWith({ showCropper: true })
    })

    it("disables showExtender when showCropper is enabled", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ 
        updateSettings,
        settings: { ...defaultSettings, showCropper: true, showExtender: true }
      })
      render(<DiffusionOptions />)
      const cropperSwitch = screen.getByLabelText(/cropper/i)
      await userEvent.click(cropperSwitch)
      expect(updateSettings).toHaveBeenCalledWith({ showCropper: true })
    })
  })

  describe("Extender", () => {
    it("toggles showExtender when clicked", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ updateSettings })
      render(<DiffusionOptions />)
      const extenderSwitch = screen.getByLabelText(/extender/i)
      await userEvent.click(extenderSwitch)
      expect(updateSettings).toHaveBeenCalledWith({ showExtender: true })
    })

    it("disables showCropper when showExtender is enabled", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ 
        updateSettings,
        settings: { ...defaultSettings, showCropper: true, showExtender: false }
      })
      render(<DiffusionOptions />)
      const extenderSwitch = screen.getByLabelText(/extender/i)
      await userEvent.click(extenderSwitch)
      expect(updateSettings).toHaveBeenCalledWith({ showExtender: true })
    })

    it("renders direction select", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/select axis/i)).toBeTruthy()
    })

    it("renders scale buttons", () => {
      render(<DiffusionOptions />)
      expect(screen.getByText(/1\.25x/i)).toBeTruthy()
      expect(screen.getByText(/1\.5x/i)).toBeTruthy()
      expect(screen.getByText(/1\.75x/i)).toBeTruthy()
      expect(screen.getByText(/2\.0x/i)).toBeTruthy()
    })
  })

  describe("BrushNet settings", () => {
    it("renders BrushNet section when model supports it", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/brushnet/i)).toBeTruthy()
    })

    it("does not render BrushNet when model does not support it", () => {
      useStore.setState({
        settings: { ...defaultSettings, model: { ...defaultSettings.model, support_brushnet: false } }
      })
      render(<DiffusionOptions />)
      expect(screen.queryByLabelText(/brushnet/i)).toBeFalsy()
    })

    it("toggles BrushNet when clicked", async () => {
      const updateEnableBrushNet = vi.fn()
      useStore.setState({ updateEnableBrushNet })
      render(<DiffusionOptions />)
      const brushnetSwitch = screen.getByLabelText(/brushnet/i)
      await userEvent.click(brushnetSwitch)
      expect(updateEnableBrushNet).toHaveBeenCalledWith(true)
    })

    it("disables BrushNet settings when switch is off", () => {
      useStore.setState({
        settings: { ...defaultSettings, enableBrushNet: false }
      })
      render(<DiffusionOptions />)
      const brushnetSlider = screen.getByLabelText(/brushnet.*weight/i)
      expect(brushnetSlider).toBeDisabled()
    })
  })

  describe("ControlNet settings", () => {
    it("renders ControlNet section when model supports it", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/controlnet/i)).toBeTruthy()
    })

    it("does not render ControlNet when model does not support it", () => {
      useStore.setState({
        settings: { ...defaultSettings, model: { ...defaultSettings.model, support_controlnet: false } }
      })
      render(<DiffusionOptions />)
      expect(screen.queryByLabelText(/controlnet/i)).toBeFalsy()
    })

    it("toggles ControlNet when clicked", async () => {
      const updateEnableControlnet = vi.fn()
      useStore.setState({ updateEnableControlnet })
      render(<DiffusionOptions />)
      const controlnetSwitch = screen.getByLabelText(/controlnet/i)
      await userEvent.click(controlnetSwitch)
      expect(updateEnableControlnet).toHaveBeenCalledWith(true)
    })
  })

  describe("LCM LoRA", () => {
    it("renders LCM LoRA section when model supports it", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/lcm lora/i)).toBeTruthy()
    })

    it("does not render LCM LoRA when model does not support it", () => {
      useStore.setState({
        settings: { ...defaultSettings, model: { ...defaultSettings.model, support_lcm_lora: false } }
      })
      render(<DiffusionOptions />)
      expect(screen.queryByLabelText(/lcm lora/i)).toBeFalsy()
    })

    it("toggles LCM LoRA when clicked", async () => {
      const updateLCMLora = vi.fn()
      useStore.setState({ updateLCMLora })
      render(<DiffusionOptions />)
      const lcmSwitch = screen.getByLabelText(/lcm lora/i)
      await userEvent.click(lcmSwitch)
      expect(updateLCMLora).toHaveBeenCalledWith(true)
    })
  })

  describe("Sliders", () => {
    it("updates Steps value", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ updateSettings })
      render(<DiffusionOptions />)
      const stepsSlider = screen.getByLabelText(/steps/i)
      await userEvent.click(stepsSlider)
      expect(updateSettings).toHaveBeenCalled()
    })

    it("updates Guidance value", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ updateSettings })
      render(<DiffusionOptions />)
      const guidanceSlider = screen.getByLabelText(/guidance/i)
      await userEvent.click(guidanceSlider)
      expect(updateSettings).toHaveBeenCalled()
    })

    it("updates Strength value when supported", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ updateSettings })
      render(<DiffusionOptions />)
      const strengthSlider = screen.getByLabelText(/strength/i)
      await userEvent.click(strengthSlider)
      expect(updateSettings).toHaveBeenCalled()
    })

    it("does not render Strength when model does not support it", () => {
      useStore.setState({
        settings: { ...defaultSettings, model: { ...defaultSettings.model, support_strength: false } }
      })
      render(<DiffusionOptions />)
      expect(screen.queryByLabelText(/strength/i)).toBeFalsy()
    })
  })

  describe("Mask operations", () => {
    it("renders Expand button", () => {
      render(<DiffusionOptions />)
      expect(screen.getByText(/expand/i)).toBeTruthy()
    })

    it("renders Shrink button", () => {
      render(<DiffusionOptions />)
      expect(screen.getByText(/shrink/i)).toBeTruthy()
    })

    it("renders Reverse button", () => {
      render(<DiffusionOptions />)
      expect(screen.getByText(/reverse/i)).toBeTruthy()
    })

    it("renders Clear button", () => {
      render(<DiffusionOptions />)
      expect(screen.getByText(/clear/i)).toBeTruthy()
    })

    it("calls adjustMask with expand", async () => {
      const adjustMask = vi.fn()
      useStore.setState({ adjustMask, isProcessing: false })
      render(<DiffusionOptions />)
      const expandButton = screen.getByText(/expand/i)
      await userEvent.click(expandButton)
      expect(adjustMask).toHaveBeenCalledWith("expand")
    })

    it("calls adjustMask with shrink", async () => {
      const adjustMask = vi.fn()
      useStore.setState({ adjustMask, isProcessing: false })
      render(<DiffusionOptions />)
      const shrinkButton = screen.getByText(/shrink/i)
      await userEvent.click(shrinkButton)
      expect(adjustMask).toHaveBeenCalledWith("shrink")
    })

    it("calls adjustMask with reverse", async () => {
      const adjustMask = vi.fn()
      useStore.setState({ adjustMask, isProcessing: false })
      render(<DiffusionOptions />)
      const reverseButton = screen.getByText(/reverse/i)
      await userEvent.click(reverseButton)
      expect(adjustMask).toHaveBeenCalledWith("reverse")
    })

    it("calls clearMask", async () => {
      const clearMask = vi.fn()
      useStore.setState({ clearMask, isProcessing: false })
      render(<DiffusionOptions />)
      const clearButton = screen.getByText(/clear/i)
      await userEvent.click(clearButton)
      expect(clearMask).toHaveBeenCalled()
    })
  })

  describe("Negative prompt", () => {
    it("updates negative prompt on input", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ updateSettings })
      render(<DiffusionOptions />)
      const textarea = screen.getByLabelText(/negative prompt/i)
      await userEvent.type(textarea, "test prompt")
      expect(updateSettings).toHaveBeenCalled()
    })

    it("does not render when model does not need prompt", () => {
      useStore.setState({
        settings: { ...defaultSettings, model: { ...defaultSettings.model, need_prompt: false } }
      })
      render(<DiffusionOptions />)
      expect(screen.queryByLabelText(/negative prompt/i)).toBeFalsy()
    })
  })

  describe("Match histograms", () => {
    it("renders Match histograms toggle", () => {
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/match histograms/i)).toBeTruthy()
    })

    it("toggles match histograms when clicked", async () => {
      const updateSettings = vi.fn()
      useStore.setState({ updateSettings })
      render(<DiffusionOptions />)
      const switchToggle = screen.getByLabelText(/match histograms/i)
      await userEvent.click(switchToggle)
      expect(updateSettings).toHaveBeenCalledWith({ sdMatchHistograms: true })
    })
  })

  describe("PowerPaint V2", () => {
    it("renders PowerPaint V2 section when supported", () => {
      useStore.setState({
        settings: { ...defaultSettings, model: { ...defaultSettings.model, support_powerpaint_v2: true } }
      })
      render(<DiffusionOptions />)
      expect(screen.getByLabelText(/powerpaint v2/i)).toBeTruthy()
    })

    it("does not render when not supported", () => {
      useStore.setState({
        settings: { ...defaultSettings, model: { ...defaultSettings.model, support_powerpaint_v2: false } }
      })
      render(<DiffusionOptions />)
      expect(screen.queryByLabelText(/powerpaint v2/i)).toBeFalsy()
    })
  })
})
