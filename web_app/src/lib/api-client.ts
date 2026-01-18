import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from "axios"
import { toast } from "@/components/ui/use-toast"

const SESSION_STORAGE_KEY = "iopaint_session_id"

function generateUUID(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID()
  }
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

export const API_ENDPOINT = import.meta.env.DEV
  ? import.meta.env.VITE_BACKEND + "/api/v1"
  : "/api/v1"

interface ApiErrorResponse {
  errors?: string
  message?: string
}

function formatError(error: AxiosError<ApiErrorResponse>): string {
  if (error.response) {
    const status = error.response.status
    const data = error.response.data
    if (data?.errors) {
      return data.errors
    }
    if (data?.message) {
      return data.message
    }
    return `Server error (${status})`
  }
  if (error.request) {
    return "Network error - please check your connection"
  }
  return error.message || "Unknown error"
}

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_ENDPOINT,
  headers: {
    "Content-Type": "application/json",
    "X-Session-Id": getOrCreateSessionId(),
  },
  timeout: 30000,
})

apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    config.headers["X-Session-Id"] = getOrCreateSessionId()
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<unknown>) => {
    const message = formatError(error as AxiosError<ApiErrorResponse>)

    toast({
      variant: "destructive",
      description: message,
    })

    console.error("API Error:", {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      message,
    })

    return Promise.reject(error)
  }
)

export default apiClient
