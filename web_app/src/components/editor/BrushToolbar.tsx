import { Expand, Download, Eraser, Eye, Redo, Undo } from "lucide-react"
import { IconButton } from "../ui/button"
import { Slider } from "../ui/slider"
import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from "@/lib/const"

interface BrushToolbarProps {
  baseBrushSize: number
  scale: number
  minScale: number
  panned: boolean
  undoDisabled: boolean
  redoDisabled: boolean
  isProcessing: boolean
  rendersLength: number
  extraMasksLength: number
  settings: {
    enableManualInpainting: boolean
    model: { model_type: string }
  }
  hadDrawSomething: () => boolean
  handleSliderChange: (value: number) => void
  resetZoom: () => void
  handleUndo: (ev: React.SyntheticEvent) => void
  handleRedo: (ev: React.SyntheticEvent) => void
  download: () => void
  runInpainting: () => void
  setShowOriginal: (value: React.SetStateAction<boolean>) => void
  setSliderPos: React.Dispatch<React.SetStateAction<number>>
  setShowRefBrush: (value: React.SetStateAction<boolean>) => void
}

const COMPARE_SLIDER_DURATION_MS = 300

export function BrushToolbar({
  baseBrushSize,
  scale,
  minScale,
  panned,
  undoDisabled,
  redoDisabled,
  isProcessing,
  rendersLength,
  extraMasksLength,
  settings,
  hadDrawSomething,
  handleSliderChange,
  resetZoom,
  handleUndo,
  handleRedo,
  download,
  runInpainting,
  setShowOriginal,
  setSliderPos,
  setShowRefBrush,
}: BrushToolbarProps) {
  return (
    <div className="fixed flex bottom-5 border px-4 py-2 rounded-[3rem] gap-8 items-center justify-center backdrop-filter backdrop-blur-md bg-background/70">
      <Slider
        className="w-48"
        defaultValue={[50]}
        min={MIN_BRUSH_SIZE}
        max={MAX_BRUSH_SIZE}
        step={1}
        tabIndex={-1}
        value={[baseBrushSize]}
        onValueChange={(vals) => handleSliderChange(vals[0])}
        onClick={() => setShowRefBrush(false)}
      />
      <div className="flex gap-2">
        <IconButton
          tooltip="Reset zoom & pan"
          disabled={scale === minScale && panned === false}
          onClick={resetZoom}
        >
          <Expand />
        </IconButton>
        <IconButton tooltip="Undo" onClick={handleUndo} disabled={undoDisabled}>
          <Undo />
        </IconButton>
        <IconButton tooltip="Redo" onClick={handleRedo} disabled={redoDisabled}>
          <Redo />
        </IconButton>
        <IconButton
          tooltip="Show original image"
          onPointerDown={(ev) => {
            ev.preventDefault()
            setShowOriginal(() => {
              window.setTimeout(() => {
                setSliderPos(100)
              }, 10)
              return true
            })
          }}
          onPointerUp={() => {
            window.setTimeout(() => {
              setSliderPos(0)
            }, 10)
            window.setTimeout(() => {
              setShowOriginal(false)
            }, COMPARE_SLIDER_DURATION_MS)
          }}
          disabled={rendersLength === 0}
        >
          <Eye />
        </IconButton>
        <IconButton
          tooltip="Save Image"
          disabled={!rendersLength}
          onClick={download}
        >
          <Download />
        </IconButton>

        {settings.enableManualInpainting &&
        settings.model.model_type === "inpaint" ? (
          <IconButton
            tooltip="Run Inpainting"
            disabled={isProcessing || (!hadDrawSomething() && extraMasksLength === 0)}
            onClick={() => {
              runInpainting()
            }}
          >
            <Eraser />
          </IconButton>
        ) : null}
      </div>
    </div>
  )
}
