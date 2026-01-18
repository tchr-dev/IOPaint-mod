import { lazy, Suspense, useEffect } from "react"
import { currentModel } from "@/lib/api"
import { useStore } from "@/lib/states"
import ImageSize from "./ImageSize"
import Plugins from "./Plugins"
import { InteractiveSeg } from "./InteractiveSeg"
import SidePanel from "./SidePanel"
import DiffusionProgress from "./DiffusionProgress"
import FileSelect from "./FileSelect"
import { Skeleton } from "./ui/skeleton"

const Editor = lazy(() => import("./Editor"))

const EditorSkeleton = () => (
  <div className="flex items-center justify-center w-full h-full">
    <div className="space-y-4 w-full max-w-md p-6">
      <Skeleton variant="rectangular" height={400} className="w-full" />
      <div className="flex justify-between">
        <Skeleton variant="rectangular" width={80} height={36} />
        <Skeleton variant="rectangular" width={80} height={36} />
        <Skeleton variant="rectangular" width={80} height={36} />
      </div>
    </div>
  </div>
)

const Workspace = () => {
const [file, updateSettings, setFile] = useStore((state) => [
  state.file,
  state.updateSettings,
  state.setFile,
])

  useEffect(() => {
    const fetchCurrentModel = async () => {
      const model = await currentModel()
      updateSettings({ model })
    }
    fetchCurrentModel()
  }, [])

  return (
    <>
      <div className="flex w-screen h-screen overflow-hidden">
        {/* Canvas area - flex-1 to take remaining space */}
        <main className="relative flex-1 flex items-center justify-center">
          {/* Upload overlay - centered in canvas, no padding tricks */}
          {!file && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <FileSelect
                onSelection={(f) => {
                  setFile(f)
                }}
              />
            </div>
          )}
          {file ? (
            <Suspense fallback={<EditorSkeleton />}>
              <Editor file={file} />
            </Suspense>
          ) : (
            <></>
          )}
        </main>

        {/* Side panel - collapsible */}
        <SidePanel />
      </div>

      <div className="flex gap-3 absolute top-[68px] left-[24px] items-center z-10">
        <Plugins />
        <ImageSize />
      </div>
      <InteractiveSeg />
      <DiffusionProgress />
    </>
  )
}

export default Workspace
