/**
 * OpenAI-Compatible API Client Functions
 *
 * This module provides typed API functions for interacting with the IOPaint
 * OpenAI-compatible endpoints. These endpoints support commercial image
 * generation models (DALL-E, etc.) through a unified interface.
 *
 * All functions include session ID headers for budget tracking (Epic 2).
 *
 * @module openai-api
 */

import { API_ENDPOINT, getOrCreateSessionId } from "./api"
import type {
  OpenAIModelInfo,
  OpenAICapabilities,
  RefinePromptRequest,
  RefinePromptResponse,
  GenerateImageRequest,
  CostEstimateRequest,
  CostEstimate,
  BudgetStatus,
  BudgetLimits,
  OpenAIImageSize,
  OpenAIImageQuality,
  OpenAIProvider,
} from "./types"

export const OPENAI_PROVIDER_BASE_URLS: Record<OpenAIProvider, string> = {
  server: "",
  proxyapi: "https://api.proxyapi.ru/openai/v1",
  openrouter: "https://openrouter.ai/api/v1",
}

export function resolveOpenAIBaseUrl(provider: OpenAIProvider): string | undefined {
  const url = OPENAI_PROVIDER_BASE_URLS[provider]
  return url ? url : undefined
}

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Custom error class for OpenAI API errors with additional metadata
 */
export class OpenAIApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public isRetryable: boolean,
    public errorType?: string
  ) {
    super(message)
    this.name = "OpenAIApiError"
  }
}

/**
 * Parse error response and throw appropriate error
 */
async function handleErrorResponse(res: Response): Promise<never> {
  let errorMessage: string
  let errorType: string | undefined

  try {
    const errorBody = await res.json()
    errorMessage = errorBody.detail || errorBody.message || errorBody.errors || "Unknown error"
    errorType = errorBody.error_type || errorBody.type
  } catch {
    errorMessage = await res.text() || `HTTP ${res.status} error`
  }

  // Determine if error is retryable
  const isRetryable = res.status === 429 || res.status >= 500

  throw new OpenAIApiError(errorMessage, res.status, isRetryable, errorType)
}

function withOpenAIHeaders(
  headers: Record<string, string>,
  baseUrl?: string
): Record<string, string> {
  if (baseUrl) {
    headers["X-OpenAI-Base-URL"] = baseUrl
  }
  return headers
}

// ============================================================================
// Model Operations
// ============================================================================

/**
 * Fetch available OpenAI-compatible models from the backend.
 *
 * @returns List of available models with metadata
 * @throws {OpenAIApiError} If the request fails
 *
 * @example
 * ```typescript
 * const models = await listOpenAIModels()
 * console.log(models[0].id) // "gpt-image-1"
 * ```
 */
export async function listOpenAIModels(
  baseUrl?: string
): Promise<OpenAIModelInfo[]> {
  const res = await fetch(`${API_ENDPOINT}/openai/models`, {
    method: "GET",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return data.models || data
}

/**
 * Fetch cached OpenAI-compatible models from the backend.
 */
export async function listOpenAIModelsCached(
  baseUrl?: string
): Promise<OpenAIModelInfo[]> {
  const res = await fetch(`${API_ENDPOINT}/openai/models/cached`, {
    method: "GET",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return data.models || data
}

/**
 * Refresh the cached OpenAI-compatible models from the backend.
 */
export async function refreshOpenAIModels(
  baseUrl?: string
): Promise<OpenAIModelInfo[]> {
  const res = await fetch(`${API_ENDPOINT}/openai/models/refresh`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return data.models || data
}

/**
 * Fetch normalized OpenAI image capabilities.
 */
export async function fetchOpenAICapabilities(
  baseUrl?: string
): Promise<OpenAICapabilities> {
  const res = await fetch(`${API_ENDPOINT}/openai/capabilities`, {
    method: "GET",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  const normalizeModels = (models: any[] = []) =>
    models.map((model) => ({
      id: model.id,
      apiId: model.api_id ?? model.apiId ?? model.id,
      label: model.label ?? model.id,
      sizes: model.sizes ?? [],
      qualities: model.qualities ?? [],
      defaultSize: model.default_size ?? model.defaultSize,
      defaultQuality: model.default_quality ?? model.defaultQuality,
    }))

  return {
    created: data.created,
    modes: {
      images_generate: {
        models: normalizeModels(data.modes?.images_generate?.models ?? []),
        defaultModel:
          data.modes?.images_generate?.default_model ??
          data.modes?.images_generate?.defaultModel,
      },
      images_edit: {
        models: normalizeModels(data.modes?.images_edit?.models ?? []),
        defaultModel:
          data.modes?.images_edit?.default_model ??
          data.modes?.images_edit?.defaultModel,
      },
    },
  }
}

// ============================================================================
// Prompt Refinement
// ============================================================================

/**
 * Refine a user's intent prompt using an LLM to create a more detailed,
 * image-generation-optimized prompt.
 *
 * This is a "cheap" operation (~$0.0001) that uses a small LLM model
 * to expand and improve the user's initial idea before the expensive
 * image generation step.
 *
 * @param request - The refinement request containing the original prompt
 * @returns The refined prompt along with the original
 * @throws {OpenAIApiError} If the request fails
 *
 * @example
 * ```typescript
 * const result = await refinePrompt({
 *   prompt: "a cat on a windowsill"
 * })
 * console.log(result.refinedPrompt)
 * // "A fluffy orange tabby cat perched gracefully on a sunlit windowsill,
 * //  gazing out at falling autumn leaves..."
 * ```
 */
export async function refinePrompt(
  request: RefinePromptRequest,
  baseUrl?: string
): Promise<RefinePromptResponse> {
  const res = await fetch(`${API_ENDPOINT}/openai/refine`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "Content-Type": "application/json",
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
    body: JSON.stringify({
      prompt: request.prompt,
      context: request.context,
      model: request.model,
      max_tokens: request.maxTokens,
    }),
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return {
    originalPrompt: data.original_prompt,
    refinedPrompt: data.refined_prompt,
    modelUsed: data.model_used,
  }
}

// ============================================================================
// Image Generation
// ============================================================================

/**
 * Generate an image from a text prompt using OpenAI-compatible API.
 *
 * Returns the generated image as a Blob for flexibility in handling
 * (display, save, convert to base64, etc.)
 *
 * @param request - Generation parameters (prompt, size, quality, etc.)
 * @returns The generated image as a Blob
 * @throws {OpenAIApiError} If generation fails
 *
 * @example
 * ```typescript
 * const blob = await generateImage({
 *   prompt: "A majestic lion in the African savanna",
 *   size: "1024x1024",
 *   quality: "hd"
 * })
 * const url = URL.createObjectURL(blob)
 * ```
 */
export async function generateImage(
  request: GenerateImageRequest,
  baseUrl?: string
): Promise<Blob> {
  const res = await fetch(`${API_ENDPOINT}/openai/generate`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "Content-Type": "application/json",
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
    body: JSON.stringify({
      prompt: request.prompt,
      n: request.n || 1,
      size: request.size || "1024x1024",
      quality: request.quality || "standard",
      model: request.model,
      style: request.style,
      negative_prompt: request.negativePrompt,
    }),
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.blob()
}

/**
 * Generate image and return as data URL for immediate display.
 * Convenience wrapper around generateImage().
 *
 * @param request - Generation parameters
 * @returns Data URL string (e.g., "data:image/png;base64,...")
 */
export async function generateImageAsDataUrl(
  request: GenerateImageRequest,
  baseUrl?: string
): Promise<string> {
  const blob = await generateImage(request, baseUrl)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

// ============================================================================
// Image Editing (Inpainting)
// ============================================================================

/**
 * Edit an existing image using a mask and prompt.
 * The mask defines which areas to regenerate (transparent = edit area).
 *
 * @param image - The source image as a Blob
 * @param mask - The mask image as a Blob (transparent areas will be edited)
 * @param prompt - Description of what to generate in the masked area
 * @param options - Additional generation options
 * @returns The edited image as a Blob
 * @throws {OpenAIApiError} If editing fails
 *
 * @example
 * ```typescript
 * const editedBlob = await editImage(
 *   originalImageBlob,
 *   maskBlob,
 *   "Replace with a golden retriever",
 *   { size: "1024x1024", quality: "hd" }
 * )
 * ```
 */
async function postImageEdit(
  endpoint: string,
  image: Blob,
  mask: Blob,
  prompt: string,
  options?: {
    n?: number
    size?: OpenAIImageSize
    quality?: OpenAIImageQuality
    model?: string
  },
  baseUrl?: string
): Promise<Blob> {
  const formData = new FormData()
  formData.append("image", image, "image.png")
  formData.append("mask", mask, "mask.png")
  formData.append("prompt", prompt)

  if (options?.n) formData.append("n", String(options.n))
  if (options?.size) formData.append("size", options.size)
  if (options?.quality) formData.append("quality", options.quality)
  if (options?.model) formData.append("model", options.model)

  const res = await fetch(`${API_ENDPOINT}${endpoint}`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
    body: formData,
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.blob()
}

export async function editImage(
  image: Blob,
  mask: Blob,
  prompt: string,
  options?: {
    n?: number
    size?: OpenAIImageSize
    quality?: OpenAIImageQuality
    model?: string
  },
  baseUrl?: string
): Promise<Blob> {
  return postImageEdit(
    "/openai/edit",
    image,
    mask,
    prompt,
    options,
    baseUrl
  )
}

/**
 * Edit image and return as data URL for immediate display.
 * Convenience wrapper around editImage().
 */
export async function editImageAsDataUrl(
  image: Blob,
  mask: Blob,
  prompt: string,
  options?: {
    n?: number
    size?: OpenAIImageSize
    quality?: OpenAIImageQuality
    model?: string
  },
  baseUrl?: string
): Promise<string> {
  const blob = await editImage(image, mask, prompt, options, baseUrl)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

export async function outpaintImage(
  image: Blob,
  mask: Blob,
  prompt: string,
  options?: {
    n?: number
    size?: OpenAIImageSize
    quality?: OpenAIImageQuality
    model?: string
  },
  baseUrl?: string
): Promise<Blob> {
  return postImageEdit(
    "/openai/outpaint",
    image,
    mask,
    prompt,
    options,
    baseUrl
  )
}

export async function outpaintImageAsDataUrl(
  image: Blob,
  mask: Blob,
  prompt: string,
  options?: {
    n?: number
    size?: OpenAIImageSize
    quality?: OpenAIImageQuality
    model?: string
  },
  baseUrl?: string
): Promise<string> {
  const blob = await outpaintImage(image, mask, prompt, options, baseUrl)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

export async function createVariation(
  image: Blob,
  options?: {
    n?: number
    size?: OpenAIImageSize
    model?: string
  },
  baseUrl?: string
): Promise<Blob> {
  const formData = new FormData()
  formData.append("image", image, "image.png")
  if (options?.n) formData.append("n", String(options.n))
  if (options?.size) formData.append("size", options.size)
  if (options?.model) formData.append("model", options.model)

  const res = await fetch(`${API_ENDPOINT}/openai/variations`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
    body: formData,
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.blob()
}

export async function createVariationAsDataUrl(
  image: Blob,
  options?: {
    n?: number
    size?: OpenAIImageSize
    model?: string
  },
  baseUrl?: string
): Promise<string> {
  const blob = await createVariation(image, options, baseUrl)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

// ============================================================================
// Tool Helpers (Upscale / Background Remove)
// ============================================================================

export async function upscaleImage(
  image: Blob,
  options?: {
    scale?: number
    size?: OpenAIImageSize
    model?: string
    prompt?: string
    mode?: "local" | "prompt" | "service"
  },
  baseUrl?: string
): Promise<Blob> {
  const formData = new FormData()
  formData.append("image", image, "image.png")
  if (options?.scale) formData.append("scale", String(options.scale))
  if (options?.size) formData.append("size", options.size)
  if (options?.model) formData.append("model", options.model)
  if (options?.prompt) formData.append("prompt", options.prompt)
  if (options?.mode) formData.append("mode", options.mode)

  const res = await fetch(`${API_ENDPOINT}/openai/upscale`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
    body: formData,
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.blob()
}

export async function removeBackground(
  image: Blob,
  options?: {
    prompt?: string
    model?: string
    mode?: "local" | "prompt" | "service"
  },
  baseUrl?: string
): Promise<Blob> {
  const formData = new FormData()
  formData.append("image", image, "image.png")
  if (options?.prompt) formData.append("prompt", options.prompt)
  if (options?.model) formData.append("model", options.model)
  if (options?.mode) formData.append("mode", options.mode)

  const res = await fetch(`${API_ENDPOINT}/openai/background-remove`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
    body: formData,
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.blob()
}

// ============================================================================
// Cost Estimation (Budget Safety Integration)
// ============================================================================

/**
 * Estimate the cost of an operation before executing it.
 * Used to show cost warnings and check budget limits.
 *
 * @param request - The operation parameters to estimate
 * @returns Cost estimate with tier classification
 *
 * @example
 * ```typescript
 * const estimate = await estimateGenerationCost({
 *   operation: "generate",
 *   model: "gpt-image-1",
 *   size: "1024x1024",
 *   quality: "hd"
 * })
 * if (estimate.tier === "high") {
 *   // Show confirmation dialog
 * }
 * ```
 */
export async function estimateGenerationCost(
  request: CostEstimateRequest
): Promise<CostEstimate> {
  const res = await fetch(`${API_ENDPOINT}/budget/estimate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
    },
    body: JSON.stringify(request),
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return {
    estimatedCostUsd: data.estimated_cost_usd,
    tier: data.cost_tier,
    warning: data.warning,
  }
}

/**
 * Get current budget status across all caps (daily, monthly, session).
 *
 * @returns Current budget usage and status
 *
 * @example
 * ```typescript
 * const status = await getBudgetStatus()
 * if (status.status === "blocked") {
 *   // Disable generation buttons
 * }
 * ```
 */
export async function getOpenAIBudgetStatus(): Promise<BudgetStatus> {
  const res = await fetch(`${API_ENDPOINT}/budget/status`, {
    method: "GET",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return {
    daily: {
      spentUsd: data.daily.spent_usd,
      remainingUsd: data.daily.remaining_usd,
      capUsd: data.daily.cap_usd,
      isUnlimited: data.daily.is_unlimited,
      percentageUsed: data.daily.cap_usd > 0
        ? (data.daily.spent_usd / data.daily.cap_usd) * 100
        : 0,
    },
    monthly: {
      spentUsd: data.monthly.spent_usd,
      remainingUsd: data.monthly.remaining_usd,
      capUsd: data.monthly.cap_usd,
      isUnlimited: data.monthly.is_unlimited,
      percentageUsed: data.monthly.cap_usd > 0
        ? (data.monthly.spent_usd / data.monthly.cap_usd) * 100
        : 0,
    },
    session: {
      spentUsd: data.session.spent_usd,
      remainingUsd: data.session.remaining_usd,
      capUsd: data.session.cap_usd,
      isUnlimited: data.session.is_unlimited,
      percentageUsed: data.session.cap_usd > 0
        ? (data.session.spent_usd / data.session.cap_usd) * 100
        : 0,
    },
    status: data.status,
    message: data.message,
  }
}

export async function getOpenAIBudgetLimits(): Promise<BudgetLimits> {
  const res = await fetch(`${API_ENDPOINT}/budget/limits`, {
    method: "GET",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return {
    dailyCapUsd: data.daily_cap_usd,
    monthlyCapUsd: data.monthly_cap_usd,
    sessionCapUsd: data.session_cap_usd,
  }
}

export async function updateOpenAIBudgetLimits(
  limits: BudgetLimits
): Promise<BudgetLimits> {
  const res = await fetch(`${API_ENDPOINT}/budget/limits`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
    },
    body: JSON.stringify({
      daily_cap_usd: limits.dailyCapUsd,
      monthly_cap_usd: limits.monthlyCapUsd,
      session_cap_usd: limits.sessionCapUsd,
    }),
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return {
    dailyCapUsd: data.daily_cap_usd,
    monthlyCapUsd: data.monthly_cap_usd,
    sessionCapUsd: data.session_cap_usd,
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert a File to Blob for API submission.
 */
export function fileToBlob(file: File): Blob {
  return new Blob([file], { type: file.type })
}

/**
 * Create a thumbnail from a data URL by resizing.
 * Used for history display to save storage space.
 *
 * @param dataUrl - Original image data URL
 * @param maxSize - Maximum dimension (width or height)
 * @returns Resized image data URL
 */
export async function createThumbnail(
  dataUrl: string,
  maxSize: number = 128
): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")!

      // Calculate new dimensions maintaining aspect ratio
      let width = img.width
      let height = img.height

      if (width > height) {
        if (width > maxSize) {
          height = (height * maxSize) / width
          width = maxSize
        }
      } else {
        if (height > maxSize) {
          width = (width * maxSize) / height
          height = maxSize
        }
      }

      canvas.width = width
      canvas.height = height
      ctx.drawImage(img, 0, 0, width, height)

      resolve(canvas.toDataURL("image/jpeg", 0.7))
    }
    img.src = dataUrl
  })
}

/**
 * Generate a fingerprint for deduplication.
 * Creates a hash from normalized request parameters.
 *
 * @param params - Generation parameters
 * @returns SHA-256 hash prefix (first 16 chars)
 */
export async function generateFingerprint(params: {
  prompt: string
  negativePrompt?: string
  model: string
  size: string
  quality: string
  n: number
}): Promise<string> {
  const normalized = JSON.stringify({
    prompt: params.prompt.trim().toLowerCase(),
    negativePrompt: (params.negativePrompt || "").trim().toLowerCase(),
    model: params.model,
    size: params.size,
    quality: params.quality,
    n: params.n,
  })

  const encoder = new TextEncoder()
  const data = encoder.encode(normalized)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, "0")).join("")

  return hashHex.substring(0, 16)
}

// ============================================================================
// Job Queue API (Epic 5)
// ============================================================================

export interface OpenAIJobSubmitRequest {
  tool: "generate" | "edit" | "outpaint" | "variation" | "upscale" | "background_remove"
  prompt?: string
  model?: string
  size?: OpenAIImageSize
  quality?: OpenAIImageQuality
  n?: number
  image_b64?: string
  mask_b64?: string
  scale?: number
  mode?: string
  intent?: string
  refined_prompt?: string
  negative_prompt?: string
  preset?: string
}

export async function submitOpenAIJob(
  request: OpenAIJobSubmitRequest,
  baseUrl?: string
): Promise<BackendGenerationJob> {
  const res = await fetch(`${API_ENDPOINT}/openai/jobs`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "Content-Type": "application/json",
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
    body: JSON.stringify(request),
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

export async function getOpenAIJob(
  jobId: string,
  baseUrl?: string
): Promise<BackendGenerationJob> {
  const res = await fetch(`${API_ENDPOINT}/openai/jobs/${jobId}`, {
    method: "GET",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

export async function cancelOpenAIJob(
  jobId: string,
  baseUrl?: string
): Promise<BackendGenerationJob> {
  const res = await fetch(`${API_ENDPOINT}/openai/jobs/${jobId}/cancel`, {
    method: "POST",
    headers: {
      ...withOpenAIHeaders(
        {
          "X-Session-Id": getOrCreateSessionId(),
        },
        baseUrl
      ),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

// ============================================================================
// History API (Epic 3 - Persistent Storage)
// ============================================================================

/**
 * Backend job record from the history API
 */
export interface BackendGenerationJob {
  id: string
  session_id: string
  status: string
  operation: string
  model: string
  intent?: string
  refined_prompt?: string
  negative_prompt?: string
  preset?: string
  params?: Record<string, unknown>
  fingerprint?: string
  estimated_cost_usd?: number
  actual_cost_usd?: number
  is_edit: boolean
  error_message?: string
  result_image_id?: string
  thumbnail_image_id?: string
  created_at: string
  completed_at?: string
}

/**
 * History list response from the backend
 */
export interface HistoryListResponse {
  jobs: BackendGenerationJob[]
  total: number
  limit: number
  offset: number
}

/**
 * Backend snapshot record from the history snapshots API
 */
export interface BackendHistorySnapshot {
  id: string
  session_id: string
  payload: Record<string, unknown>
  created_at: string
}

/**
 * History snapshot list response from the backend
 */
export interface HistorySnapshotListResponse {
  snapshots: BackendHistorySnapshot[]
  total: number
  limit: number
  offset: number
}

/**
 * Parameters for creating a new history entry
 */
export interface CreateHistoryRequest {
  operation: "generate" | "edit" | "refine"
  model: string
  intent?: string
  refined_prompt?: string
  negative_prompt?: string
  preset?: string
  params?: Record<string, unknown>
  fingerprint?: string
  estimated_cost_usd?: number
  is_edit?: boolean
}

/**
 * Parameters for updating a history entry
 */
export interface UpdateHistoryRequest {
  status?:
    | "queued"
    | "running"
    | "succeeded"
    | "failed"
    | "blocked_budget"
    | "cancelled"
  actual_cost_usd?: number
  error_message?: string
  result_image_id?: string
  thumbnail_image_id?: string
}

/**
 * Fetch generation history from the backend.
 *
 * @param options - Filtering and pagination options
 * @returns List of generation jobs with pagination info
 *
 * @example
 * ```typescript
 * const { jobs, total } = await fetchHistory({ status: "succeeded", limit: 20 })
 * ```
 */
export async function fetchHistory(options?: {
  status?: string
  limit?: number
  offset?: number
}): Promise<HistoryListResponse> {
  const params = new URLSearchParams()
  if (options?.status) params.set("status", options.status)
  if (options?.limit) params.set("limit", String(options.limit))
  if (options?.offset) params.set("offset", String(options.offset))

  const queryString = params.toString()
  const url = queryString
    ? `${API_ENDPOINT}/history?${queryString}`
    : `${API_ENDPOINT}/history`

  const res = await fetch(url, {
    method: "GET",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

/**
 * Get a specific history entry by ID.
 *
 * @param jobId - The job ID to retrieve
 * @returns The generation job record
 */
export async function getHistoryJob(jobId: string): Promise<BackendGenerationJob> {
  const res = await fetch(`${API_ENDPOINT}/history/${jobId}`, {
    method: "GET",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

/**
 * Create a new history entry in the backend.
 *
 * @param job - The job data to create
 * @returns The created job record with ID
 */
export async function createHistoryJob(
  job: CreateHistoryRequest
): Promise<BackendGenerationJob> {
  const res = await fetch(`${API_ENDPOINT}/history`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
    },
    body: JSON.stringify(job),
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

/**
 * Update a history entry.
 *
 * @param jobId - The job ID to update
 * @param updates - The fields to update
 * @returns The updated job record
 */
export async function updateHistoryJob(
  jobId: string,
  updates: UpdateHistoryRequest
): Promise<BackendGenerationJob> {
  const res = await fetch(`${API_ENDPOINT}/history/${jobId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
    },
    body: JSON.stringify(updates),
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

/**
 * Delete a history entry.
 *
 * @param jobId - The job ID to delete
 */
export async function deleteHistoryJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_ENDPOINT}/history/${jobId}`, {
    method: "DELETE",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }
}

/**
 * Clear all history for the current session.
 *
 * @returns The number of jobs deleted
 */
export async function clearHistory(): Promise<{ deleted: number }> {
  const res = await fetch(`${API_ENDPOINT}/history/clear`, {
    method: "DELETE",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

/**
 * Fetch history snapshots from the backend.
 */
export async function fetchHistorySnapshots(options?: {
  limit?: number
  offset?: number
}): Promise<HistorySnapshotListResponse> {
  const params = new URLSearchParams()
  if (options?.limit) params.set("limit", String(options.limit))
  if (options?.offset) params.set("offset", String(options.offset))

  const queryString = params.toString()
  const url = queryString
    ? `${API_ENDPOINT}/history/snapshots?${queryString}`
    : `${API_ENDPOINT}/history/snapshots`

  const res = await fetch(url, {
    method: "GET",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

/**
 * Create a history snapshot in the backend.
 */
export async function createHistorySnapshot(payload: Record<string, unknown>): Promise<BackendHistorySnapshot> {
  const res = await fetch(`${API_ENDPOINT}/history/snapshots`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
    },
    body: JSON.stringify({ payload }),
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

/**
 * Get a specific history snapshot by ID.
 */
export async function getHistorySnapshot(snapshotId: string): Promise<BackendHistorySnapshot> {
  const res = await fetch(`${API_ENDPOINT}/history/snapshots/${snapshotId}`, {
    method: "GET",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

/**
 * Delete a history snapshot.
 */
export async function deleteHistorySnapshot(snapshotId: string): Promise<void> {
  const res = await fetch(`${API_ENDPOINT}/history/snapshots/${snapshotId}`, {
    method: "DELETE",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }
}

/**
 * Clear all history snapshots for the current session.
 */
export async function clearHistorySnapshots(): Promise<{ deleted: number }> {
  const res = await fetch(`${API_ENDPOINT}/history/snapshots/clear`, {
    method: "DELETE",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.json()
}

// ============================================================================
// Image Storage API (Epic 3)
// ============================================================================

/**
 * Get the URL for a stored image.
 *
 * @param imageId - The image ID
 * @returns Full URL to fetch the image
 */
export function getStoredImageUrl(imageId: string): string {
  return `${API_ENDPOINT}/storage/images/${imageId}`
}

/**
 * Get the URL for an image thumbnail.
 *
 * @param imageId - The image ID
 * @returns Full URL to fetch the thumbnail
 */
export function getStoredThumbnailUrl(imageId: string): string {
  return `${API_ENDPOINT}/storage/images/${imageId}/thumbnail`
}

/**
 * Fetch a stored image as a Blob.
 *
 * @param imageId - The image ID
 * @returns The image as a Blob
 */
export async function fetchStoredImage(imageId: string): Promise<Blob> {
  const res = await fetch(getStoredImageUrl(imageId), {
    method: "GET",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.blob()
}

/**
 * Fetch a stored image as a data URL for display.
 *
 * @param imageId - The image ID
 * @returns The image as a data URL string
 */
export async function fetchStoredImageAsDataUrl(imageId: string): Promise<string> {
  const blob = await fetchStoredImage(imageId)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}
