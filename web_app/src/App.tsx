import { useEffect, useRef } from "react"

import useInputImage from "@/hooks/useInputImage"
import { keepGUIAlive } from "@/lib/utils"
import { getServerConfig } from "@/lib/api"
import Header from "@/components/Header"
import Workspace from "@/components/Workspace"
import { ErrorBoundary } from "./components/ui/error-boundary"
import { Toaster } from "./components/ui/toaster"
import { useStore } from "./lib/states"
import { useWindowSize } from "react-use"

const SUPPORTED_FILE_TYPE = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
  "image/tiff",
]

import { BrowserRouter, Routes, Route } from "react-router-dom"
import Settings from "@/pages/Settings"

function Home() {
  const [updateAppState, setServerConfig, setFile] = useStore((state) => [
    state.updateAppState,
    state.setServerConfig,
    state.setFile,
  ])

  const userInputImage = useInputImage()
  const windowSize = useWindowSize()
  const dragCounter = useRef(0)

  useEffect(() => {
    if (userInputImage) {
      setFile(userInputImage)
    }
  }, [userInputImage, setFile])

  useEffect(() => {
    updateAppState({ windowSize })
  }, [windowSize, updateAppState])

  useEffect(() => {
    const fetchServerConfig = async () => {
      const serverConfig = await getServerConfig()
      setServerConfig(serverConfig)
      if (serverConfig.isDesktop) {
        keepGUIAlive()
      }
    }
    fetchServerConfig()
  }, [setServerConfig])

  useEffect(() => {
    const onDragEnter = () => {
      dragCounter.current += 1
    }
    const onDragLeave = () => {
      dragCounter.current -= 1
      if (dragCounter.current > 0) return
    }
    const onDragOver = (event: DragEvent) => {
      event.preventDefault()
      event.stopPropagation()
    }
    const onDrop = (event: DragEvent) => {
      event.preventDefault()
      event.stopPropagation()
      const dataTransfer = event.dataTransfer
      if (dataTransfer && dataTransfer.files && dataTransfer.files.length > 0) {
        if (dataTransfer.files.length > 1) {
          // Multiple files not supported
        } else {
          const dragFile = dataTransfer.files[0]
          const fileType = dragFile.type
          if (SUPPORTED_FILE_TYPE.includes(fileType)) {
            setFile(dragFile)
          }
        }
        dataTransfer.clearData()
      }
    }
    const onPaste = (event: ClipboardEvent) => {
      if (!event.clipboardData) {
        return
      }
      const clipboardItems = event.clipboardData.items
      const items: DataTransferItem[] = [].slice
        .call(clipboardItems)
        .filter((item: DataTransferItem) => {
          return item.type.indexOf("image") !== -1
        })

      if (items.length === 0) {
        return
      }

      event.preventDefault()
      event.stopPropagation()

      const item = items[0]
      const blob = item.getAsFile()
      if (blob) {
        setFile(blob)
      }
    }

    window.addEventListener("dragenter", onDragEnter)
    window.addEventListener("dragleave", onDragLeave)
    window.addEventListener("dragover", onDragOver)
    window.addEventListener("drop", onDrop)
    window.addEventListener("paste", onPaste)
    return function cleanUp() {
      window.removeEventListener("dragenter", onDragEnter)
      window.removeEventListener("dragleave", onDragLeave)
      window.removeEventListener("dragover", onDragOver)
      window.removeEventListener("drop", onDrop)
      window.removeEventListener("paste", onPaste)
    }
  }, [setFile])

  return (
    <ErrorBoundary
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-background">
          <div className="text-center p-6">
            <h2 className="text-lg font-semibold mb-2">Something went wrong</h2>
            <p className="text-muted-foreground mb-4">
              Please refresh the page to continue
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-md"
            >
              Refresh Page
            </button>
          </div>
        </div>
      }
    >
      <main className="flex min-h-screen flex-col items-center justify-between w-full bg-[radial-gradient(circle_at_1px_1px,_#8e8e8e8e_1px,_transparent_0)] [background-size:20px_20px] bg-repeat">
        <Toaster />
        <Header />
        <Workspace />
      </main>
    </ErrorBoundary>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
