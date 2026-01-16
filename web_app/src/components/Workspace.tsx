import { useEffect } from "react"
import Editor from "./Editor"
import { currentModel } from "@/lib/api"
import { useStore } from "@/lib/states"
import ImageSize from "./ImageSize"
import Plugins from "./Plugins"
import { InteractiveSeg } from "./InteractiveSeg"
import SidePanel from "./SidePanel"
import DiffusionProgress from "./DiffusionProgress"
import FileSelect from "./FileSelect"

const Workspace = () => {
  const [file, updateSettings] = useStore((state) => [
    state.file,
    state.updateSettings,
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
                  // File selection handled via store
                }}
              />
            </div>
          )}
          {file ? <Editor file={file} /> : <></>}
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
