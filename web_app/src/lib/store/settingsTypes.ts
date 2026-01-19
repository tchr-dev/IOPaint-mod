import type { ModelInfo, OpenAIProvider, OpenAIToolMode } from "../types"
import { ExtenderDirection, LDMSampler, CV2Flag, PowerPaintTask } from "../types"
import { QualityPreset } from "../presets"

export type Settings = {
  model: ModelInfo
  qualityPreset: QualityPreset
  enableDownloadMask: boolean
  enableManualInpainting: boolean
  enableUploadMask: boolean
  enableAutoExtractPrompt: boolean
  openAIProvider: OpenAIProvider
  openAIToolMode: OpenAIToolMode
  showCropper: boolean
  showExtender: boolean
  extenderDirection: ExtenderDirection
  ldmSteps: number
  ldmSampler: LDMSampler
  zitsWireframe: boolean
  cv2Radius: number
  cv2Flag: CV2Flag
  prompt: string
  negativePrompt: string
  seed: number
  seedFixed: boolean
  sdMaskBlur: number
  sdStrength: number
  sdSteps: number
  sdGuidanceScale: number
  sdSampler: string
  sdMatchHistograms: boolean
  sdScale: number
  p2pImageGuidanceScale: number
  enableControlnet: boolean
  controlnetConditioningScale: number
  controlnetMethod: string
  enableBrushNet: boolean
  brushnetMethod: string
  brushnetConditioningScale: number
  enableLCMLora: boolean
  enablePowerPaintV2: boolean
  powerpaintTask: PowerPaintTask
  adjustMaskKernelSize: number
}

export const settingsDefaultValues: Settings = {
  model: {
    name: "lama",
    path: "lama",
    model_type: "inpaint",
    support_controlnet: false,
    support_brushnet: false,
    support_strength: false,
    support_outpainting: false,
    support_powerpaint_v2: false,
    controlnets: [],
    brushnets: [],
    support_lcm_lora: false,
    is_single_file_diffusers: false,
    need_prompt: false,
  },
  qualityPreset: QualityPreset.FAST,
  showCropper: false,
  showExtender: false,
  extenderDirection: ExtenderDirection.xy,
  enableDownloadMask: false,
  enableManualInpainting: false,
  enableUploadMask: false,
  enableAutoExtractPrompt: true,
  openAIProvider: "server",
  openAIToolMode: "local",
  ldmSteps: 30,
  ldmSampler: LDMSampler.ddim,
  zitsWireframe: true,
  cv2Radius: 5,
  cv2Flag: CV2Flag.INPAINT_NS,
  prompt: "",
  negativePrompt: "bad quality, worse anatomy, blur, noisy",
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
  powerpaintTask: PowerPaintTask.text_guided,
  adjustMaskKernelSize: 12,
}
