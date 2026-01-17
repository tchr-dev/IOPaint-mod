import {
  Filename,
  GenInfo,
  ModelInfo,
  PowerPaintTask,
  Rect,
  ServerConfig,
} from "@/lib/types"
import { Settings } from "@/lib/states"
import { convertToBase64, srcToFile } from "@/lib/utils"
import axios from "axios"

export const API_ENDPOINT = import.meta.env.DEV
  ? import.meta.env.VITE_BACKEND + "/api/v1"
  : "/api/v1"

export const inpaintController = new AbortController()

// Session ID management for budget tracking
const SESSION_STORAGE_KEY = "iopaint_session_id"

function generateUUID(): string {
  // Use crypto.randomUUID if available, fallback to manual generation
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  // Fallback for older browsers
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === "x" ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

export function getOrCreateSessionId(): string {
  let sessionId = localStorage.getItem(SESSION_STORAGE_KEY)
  if (!sessionId) {
    sessionId = generateUUID()
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId)
  }
  return sessionId
}

export function resetSessionId(): string {
  const newSessionId = generateUUID()
  localStorage.setItem(SESSION_STORAGE_KEY, newSessionId)
  return newSessionId
}

const api = axios.create({
  baseURL: API_ENDPOINT,
  headers: {
    "X-Session-Id": getOrCreateSessionId(),
  },
})

const throwErrors = async (res: any): Promise<never> => {
  const errMsg = await res.json()
  throw new Error(
    `${errMsg.errors}\nPlease take a screenshot of the detailed error message in your terminal`
  )
}

export default async function inpaint(
  imageFile: File,
  settings: Settings,
  croperRect: Rect,
  extenderState: Rect,
  mask: File | Blob,
  paintByExampleImage: File | null = null,
  signal?: AbortSignal
) {
  const imageBase64 = await convertToBase64(imageFile)
  const maskBase64 = await convertToBase64(mask)
  const exampleImageBase64 = paintByExampleImage
    ? await convertToBase64(paintByExampleImage)
    : null

  const res = await fetch(`${API_ENDPOINT}/inpaint`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
    },
    signal,
    body: JSON.stringify({
      image: imageBase64,
      mask: maskBase64,
      ldm_steps: settings.ldmSteps,
      ldm_sampler: settings.ldmSampler,
      zits_wireframe: settings.zitsWireframe,
      cv2_flag: settings.cv2Flag,
      cv2_radius: settings.cv2Radius,
      hd_strategy: "Crop",
      hd_strategy_crop_triger_size: 640,
      hd_strategy_crop_margin: 128,
      hd_trategy_resize_imit: 2048,
      prompt: settings.prompt,
      negative_prompt: settings.negativePrompt,
      use_croper: settings.showCropper,
      croper_x: croperRect.x,
      croper_y: croperRect.y,
      croper_height: croperRect.height,
      croper_width: croperRect.width,
      use_extender: settings.showExtender,
      extender_x: extenderState.x,
      extender_y: extenderState.y,
      extender_height: extenderState.height,
      extender_width: extenderState.width,
      sd_mask_blur: settings.sdMaskBlur,
      sd_strength: settings.sdStrength,
      sd_steps: settings.sdSteps,
      sd_guidance_scale: settings.sdGuidanceScale,
      sd_sampler: settings.sdSampler,
      sd_seed: settings.seedFixed ? settings.seed : -1,
      sd_match_histograms: settings.sdMatchHistograms,
      sd_lcm_lora: settings.enableLCMLora,
      paint_by_example_example_image: exampleImageBase64,
      p2p_image_guidance_scale: settings.p2pImageGuidanceScale,
      enable_controlnet: settings.enableControlnet,
      controlnet_conditioning_scale: settings.controlnetConditioningScale,
      controlnet_method: settings.controlnetMethod
        ? settings.controlnetMethod
        : "",
      enable_brushnet: settings.enableBrushNet,
      brushnet_method: settings.brushnetMethod ? settings.brushnetMethod : "",
      brushnet_conditioning_scale: settings.brushnetConditioningScale,
      enable_powerpaint_v2: settings.enablePowerPaintV2,
      powerpaint_task: settings.showExtender
        ? PowerPaintTask.outpainting
        : settings.powerpaintTask,
    }),
  })
  if (res.ok) {
    const blob = await res.blob()
    return {
      blob: URL.createObjectURL(blob),
      seed: res.headers.get("X-Seed"),
    }
  }
  throw await throwErrors(res)
}

export async function getServerConfig(): Promise<ServerConfig> {
  const res = await api.get(`/server-config`)
  return res.data
}

export async function switchModel(name: string): Promise<ModelInfo> {
  const res = await api.post(`/model`, { name })
  return res.data
}

export async function switchPluginModel(
  plugin_name: string,
  model_name: string
) {
  return api.post(`/switch_plugin_model`, { plugin_name, model_name })
}

export async function currentModel(): Promise<ModelInfo> {
  const res = await api.get("/model")
  return res.data
}

export async function runPlugin(
  genMask: boolean,
  name: string,
  imageFile: File,
  upscale?: number,
  clicks?: number[][]
) {
  const imageBase64 = await convertToBase64(imageFile)
  const p = genMask ? "run_plugin_gen_mask" : "run_plugin_gen_image"
  const res = await fetch(`${API_ENDPOINT}/${p}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
    },
    body: JSON.stringify({
      name,
      image: imageBase64,
      scale: upscale,
      clicks,
    }),
  })
  if (res.ok) {
    const blob = await res.blob()
    return { blob: URL.createObjectURL(blob) }
  }
  throw await throwErrors(res)
}

export async function getMediaFile(tab: string, filename: string) {
  const res = await fetch(
    `${API_ENDPOINT}/media_file?tab=${tab}&filename=${encodeURIComponent(
      filename
    )}`,
    {
      method: "GET",
    }
  )
  if (res.ok) {
    const blob = await res.blob()
    const file = new File([blob], filename, {
      type: res.headers.get("Content-Type") ?? "image/png",
    })
    return file
  }
  throw await throwErrors(res)
}

export async function getMediaBlob(tab: string, filename: string) {
  const res = await fetch(
    `${API_ENDPOINT}/media_file?tab=${tab}&filename=${encodeURIComponent(
      filename
    )}`,
    {
      method: "GET",
    }
  )
  if (res.ok) {
    const blob = await res.blob()
    return blob
  }
  throw await throwErrors(res)
}

export async function getMedias(tab: string): Promise<Filename[]> {
  const res = await api.get(`medias`, { params: { tab } })
  return res.data
}

export async function downloadToOutput(
  image: HTMLImageElement,
  filename: string,
  mimeType: string
) {
  const file = await srcToFile(image.src, filename, mimeType)
  const fd = new FormData()
  fd.append("file", file)

  try {
    const res = await fetch(`${API_ENDPOINT}/save_image`, {
      method: "POST",
      body: fd,
    })
    if (!res.ok) {
      throw await throwErrors(res)
    }
  } catch (error) {
    throw new Error(`Something went wrong: ${error}`)
  }
}

export async function getGenInfo(file: File): Promise<GenInfo> {
  const fd = new FormData()
  fd.append("file", file)
  const res = await api.post(`/gen-info`, fd)
  return res.data
}

export async function getSamplers(): Promise<string[]> {
  const res = await api.post("/samplers")
  return res.data
}

export async function postAdjustMask(
  mask: File | Blob,
  operate: "expand" | "shrink" | "reverse",
  kernel_size: number
) {
  const maskBase64 = await convertToBase64(mask)
  const res = await fetch(`${API_ENDPOINT}/adjust_mask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
    },
    body: JSON.stringify({
      mask: maskBase64,
      operate: operate,
      kernel_size: kernel_size,
    }),
  })
  if (res.ok) {
    const blob = await res.blob()
    return blob
  }
  throw await throwErrors(res)
}

// Budget Safety API types
export interface BudgetUsage {
  spent_usd: number
  remaining_usd: number
  cap_usd: number
  is_unlimited: boolean
}

export interface BudgetStatusResponse {
  daily: BudgetUsage
  monthly: BudgetUsage
  session: BudgetUsage
  status: "ok" | "warning" | "blocked"
  message: string | null
}

export interface CostEstimateRequest {
  operation: "generate" | "edit" | "variation" | "refine"
  model: string
  size?: string
  quality?: string
  n?: number
}

export interface CostEstimateResponse {
  estimated_cost_usd: number
  cost_tier: "low" | "medium" | "high"
  warning: string | null
}

// Budget Safety API functions
export async function getBudgetStatus(): Promise<BudgetStatusResponse> {
  const res = await api.get("/budget/status")
  return res.data
}

export async function estimateCost(
  params: CostEstimateRequest
): Promise<CostEstimateResponse> {
  const res = await api.post("/budget/estimate", params)
  return res.data
}
