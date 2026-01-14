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
  RefinePromptRequest,
  RefinePromptResponse,
  GenerateImageRequest,
  CostEstimateRequest,
  CostEstimate,
  BudgetStatus,
  OpenAIImageSize,
  OpenAIImageQuality,
} from "./types"

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
export async function listOpenAIModels(): Promise<OpenAIModelInfo[]> {
  const res = await fetch(`${API_ENDPOINT}/openai/models`, {
    method: "GET",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  const data = await res.json()
  return data.models || data
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
  request: RefinePromptRequest
): Promise<RefinePromptResponse> {
  const res = await fetch(`${API_ENDPOINT}/openai/refine`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
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
  request: GenerateImageRequest
): Promise<Blob> {
  const res = await fetch(`${API_ENDPOINT}/openai/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getOrCreateSessionId(),
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
  request: GenerateImageRequest
): Promise<string> {
  const blob = await generateImage(request)
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
export async function editImage(
  image: Blob,
  mask: Blob,
  prompt: string,
  options?: {
    n?: number
    size?: OpenAIImageSize
    quality?: OpenAIImageQuality
    model?: string
  }
): Promise<Blob> {
  const formData = new FormData()
  formData.append("image", image, "image.png")
  formData.append("mask", mask, "mask.png")
  formData.append("prompt", prompt)

  if (options?.n) formData.append("n", String(options.n))
  if (options?.size) formData.append("size", options.size)
  if (options?.quality) formData.append("quality", options.quality)
  if (options?.model) formData.append("model", options.model)

  const res = await fetch(`${API_ENDPOINT}/openai/edit`, {
    method: "POST",
    headers: {
      "X-Session-Id": getOrCreateSessionId(),
    },
    body: formData,
  })

  if (!res.ok) {
    await handleErrorResponse(res)
  }

  return res.blob()
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
  }
): Promise<string> {
  const blob = await editImage(image, mask, prompt, options)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
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
