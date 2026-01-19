export interface Filename {
  name: string
  height: number
  width: number
  ctime: number
  mtime: number
}

export interface PluginInfo {
  name: string
  support_gen_image: boolean
  support_gen_mask: boolean
}


export interface ApiConfig {
  host: string
  port: number
  inbrowser: boolean
  model: string
  no_half: boolean
  low_mem: boolean
  cpu_offload: boolean
  disable_nsfw_checker: boolean
  local_files_only: boolean
  cpu_textencoder: boolean
  device: string
  input?: string
  mask_dir?: string
  output_dir?: string
  quality: number
  enable_interactive_seg: boolean
  interactive_seg_model: string
  interactive_seg_device: string
  enable_remove_bg: boolean
  remove_bg_device: string
  remove_bg_model: string
  enable_anime_seg: boolean
  enable_realesrgan: boolean
  realesrgan_device: string
  realesrgan_model: string
  enable_gfpgan: boolean
  gfpgan_device: string
  enable_restoreformer: boolean
  restoreformer_device: string
  enable_file_manager: boolean
  enable_auto_saving: boolean
  enable_controlnet: boolean
  controlnet_method?: string
  disable_model_switch: boolean
  is_desktop: boolean
}

export interface ServerConfig {
  plugins: PluginInfo[]
  modelInfos: ModelInfo[]
  removeBGModel: string
  removeBGModels: string[]
  realesrganModel: string
  realesrganModels: string[]
  interactiveSegModel: string
  interactiveSegModels: string[]
  enableFileManager: boolean
  enableAutoSaving: boolean
  enableControlnet: boolean
  controlnetMethod: string
  disableModelSwitch: boolean
  isDesktop: boolean
  samplers: string[]
}

export interface GenInfo {
  prompt: string
  negative_prompt: string
}

export interface ModelInfo {
  name: string
  path: string
  model_type:
  | "inpaint"
  | "diffusers_sd"
  | "diffusers_sdxl"
  | "diffusers_sd_inpaint"
  | "diffusers_sdxl_inpaint"
  | "diffusers_other"
  support_strength: boolean
  support_outpainting: boolean
  support_controlnet: boolean
  support_brushnet: boolean
  support_powerpaint_v2: boolean
  controlnets: string[]
  brushnets: string[]
  support_lcm_lora: boolean
  need_prompt: boolean
  is_single_file_diffusers: boolean
}

export enum SortBy {
  NAME = "name",
  CTIME = "ctime",
  MTIME = "mtime",
}

export enum SortOrder {
  DESCENDING = "desc",
  ASCENDING = "asc",
}

export enum LDMSampler {
  ddim = "ddim",
  plms = "plms",
}

export enum CV2Flag {
  INPAINT_NS = "INPAINT_NS",
  INPAINT_TELEA = "INPAINT_TELEA",
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface Point {
  x: number
  y: number
}

export interface Line {
  size?: number
  pts: Point[]
}

export type LineGroup = Array<Line>

export interface Size {
  width: number
  height: number
}

export interface StoredImage {
  id: string
  src: string
  width: number
  height: number
}

export enum ExtenderDirection {
  x = "x",
  y = "y",
  xy = "xy",
}

export enum PowerPaintTask {
  text_guided = "text-guided",
  shape_guided = "shape-guided",
  context_aware = "context-aware",
  object_remove = "object-remove",
  outpainting = "outpainting",
}

export type AdjustMaskOperate = "expand" | "shrink" | "reverse"

